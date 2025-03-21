import os
import asyncio
import requests
import random 
import concurrent
import aiohttp
import time
import logging
from typing import List, Optional, Dict, Any, Union
from urllib.parse import unquote

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup

from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langsmith import traceable

from open_deep_research.state import Section


def get_config_value(value):
    """設定値を適切な形式で取得するヘルパー関数
    
    この関数は、設定値が文字列かEnum型かを判断し、適切な形式で値を返します。
    LangGraphのコンフィグでは、設定値が以下の2つの形式で渡される可能性があります：
    1. 文字列形式："perplexity", "tavily"など
    2. Enum型オブジェクト：SearchAPI.PERPLEXITY, SearchAPI.TAVILYなど
    
    この関数により、どちらの形式で設定が渡されても一貫した方法で処理できます。
    
    Args:
        value: 設定値（文字列またはEnum）
        
    Returns:
        str: 文字列形式の設定値
    """
    return value if isinstance(value, str) else value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """検索API固有のパラメータをフィルタリングするヘルパー関数
    
    この関数は、異なる検索APIが異なるパラメータをサポートすることに対応するため、
    指定されたAPIがサポートするパラメータだけをフィルタリングして返します。
    
    例えば、Exaは「max_characters」「include_domains」などをサポートしますが、
    Tavilyはこれらのパラメータをサポートしていません。この関数は、指定されたAPIに
    対応するパラメータだけを含む設定辞書を生成します。
    
    Args:
        search_api (str): 検索APIの識別子（例："exa", "tavily"）
        search_api_config (Optional[Dict[str, Any]]): 検索API用の設定辞書
        
    Returns:
        Dict[str, Any]: 検索関数に渡すパラメータの辞書
    """
    # 各検索APIがサポートするパラメータを定義
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": [],  # Tavilyは現在追加パラメータをサポートしていない
        "perplexity": [],  # Perplexityも追加パラメータをサポートしていない
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
    }

    # 指定された検索APIがサポートするパラメータのリストを取得
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # 設定が提供されていない場合は空の辞書を返す
    if not search_api_config:
        return {}

    # 設定をフィルタリングして、サポートされているパラメータのみを含める
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """検索結果の重複を排除し、整形された文字列に変換する関数
    
    この関数は、複数の検索クエリから得られた結果を処理し、以下の操作を行います：
    1. すべての検索結果を一つのリストに統合
    2. URLを基準に重複するソースを排除（同じページが複数のクエリでヒットする場合）
    3. 各ソースを限られたトークン数に制限（トークン消費を抑制するため）
    4. 整形されたテキストとして返却（AIモデルが読みやすい形式）
    
    Args:
        search_response (List[Dict]): 検索APIからの応答のリスト。各応答は以下の構造:
            - query: str (検索クエリ)
            - results: List[Dict] (検索結果のリスト)
                - title: str (ページタイトル)
                - url: str (ページURL)
                - content: str (抜粋・要約)
                - score: float (関連性スコア)
                - raw_content: str|None (ページの生コンテンツ)
        max_tokens_per_source (int): 各ソースから取得する最大トークン数（概算）
        include_raw_content (bool): 生のコンテンツを含めるかどうか。デフォルトはTrue
            
    Returns:
        str: 重複排除・整形済みの検索結果テキスト
    """
    # 1. すべての検索結果を一つのリストに統合
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
    
    # 2. URLを基準に重複を排除
    # 同じURLのソースがある場合、辞書内では後のエントリが前のエントリを上書き
    unique_sources = {source['url']: source for source in sources_list}

    # 3. 結果を整形して文字列として出力
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        # 3.1 セクション区切り
        formatted_text += f"{'='*80}\n"  # 明確なセクション区切り
        
        # 3.2 ソースのタイトルとURL
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # サブセクション区切り
        formatted_text += f"URL: {source['url']}\n===\n"
        
        # 3.3 関連する抜粋・要約
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        
        # 3.4 生コンテンツ（オプション）
        if include_raw_content:
            # トークン数を概算（1トークン≒4文字と仮定）
            char_limit = max_tokens_per_source * 4
            
            # raw_contentがない場合の処理
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            
            # 長すぎる内容を切り詰め
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
                
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        
        # 3.5 セクション終了
        formatted_text += f"{'='*80}\n\n" # セクション終了区切り
                
    # 4. 整形されたテキストの末尾の空白を削除して返却
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """セクションのリストを整形された文字列に変換する関数
    
    この関数は、レポートの各セクションを視覚的に区別しやすい形式に整形します。
    各セクションは区切り線で明確に分けられ、セクション番号、名前、説明、
    調査が必要かどうか、および内容を含みます。
    
    主に以下の用途で使用されます：
    1. 最終セクション（はじめに/結論など）執筆時のコンテキスト提供
    2. デバッグやログ出力用のレポート内容の可視化
    
    Args:
        sections (list[Section]): Section型オブジェクトのリスト。
            各Sectionは名前、説明、調査要否、内容などの属性を持つ
        
    Returns:
        str: 整形されたセクション内容の文字列
    """
    # 整形文字列の初期化
    formatted_str = ""
    
    # 各セクションを順番に処理（1から始まる番号付け）
    for idx, section in enumerate(sections, 1):
        # セクションの整形テンプレート
        formatted_str += f"""
        {'='*60}
        Section {idx}: {section.name}
        {'='*60}
        Description:
        {section.description}
        Requires Research: 
        {section.research}
        Content:
        {section.content if section.content else '[Not yet written]'}
        """
    # 整形された文字列を返却
    return formatted_str

@traceable
async def tavily_search_async(search_queries):
    """
    Tavily APIを使用して並行的にWeb検索を実行します。

    Args:
        search_queries (List[SearchQuery]): 処理する検索クエリのリスト

    Returns:
            List[dict]: Tavily APIからの検索応答のリスト（クエリごとに1つ）。各応答は以下の形式です：
                {
                    'query': str, # 元の検索クエリ
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # 検索結果のリスト
                        {
                            'title': str,            # ウェブページのタイトル
                            'url': str,              # 結果のURL
                            'content': str,          # コンテンツの要約/スニペット
                            'score': float,          # 関連性スコア
                            'raw_content': str|None  # 利用可能な場合はページの全内容
                        },
                        ...
                    ]
                }
    """
    # TavilyのAPIを非同期で利用するためのクライアントを初期化
    tavily_async_client = AsyncTavilyClient()
    
    # 各検索クエリに対応する非同期タスクのリストを作成するための空リスト
    search_tasks = []
    
    # 各クエリに対して非同期検索タスクを作成し、リストに追加
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=5,                # 検索結果は最大5件まで取得
                    include_raw_content=True,     # 生のコンテンツ（ウェブページの全文）も含める
                    topic="general"               # 一般的なトピックで検索
                )
            )

    # asyncio.gatherを使用してすべての検索タスクを並行実行
    # これにより単一のクエリを順番に処理するより大幅に高速化される
    search_docs = await asyncio.gather(*search_tasks)

    # 全検索結果のリストを返却
    # 各要素は1つの検索クエリに対応するAPIレスポンスの辞書
    return search_docs

@traceable
def perplexity_search(search_queries):
    """Perplexity APIを使用してウェブ検索を実行する関数
    
    この関数は、Perplexity AIのチャット補完APIを使用して検索を行います。
    Perplexityは高度な検索エンジンとLLM技術を組み合わせたサービスで、
    検索クエリに対して、事実に基づいた回答とその情報源を提供します。
    
    特徴:
    1. 通常の検索エンジンと異なり、構造化された回答を生成
    2. 回答の根拠となる情報源（URL）を提供
    3. 他の検索APIと互換性のある形式で結果を返却
    
    Args:
        search_queries (List[str]): 検索クエリの文字列リスト
  
    Returns:
        List[dict]: 検索応答のリスト。各応答は以下の形式:
            {
                'query': str,                    # 元の検索クエリ
                'follow_up_questions': None,     # 後続質問（この実装では未使用）
                'answer': None,                  # 回答（この実装では未使用）
                'images': list,                  # 画像リスト（通常は空）
                'results': [                     # 検索結果のリスト
                    {
                        'title': str,            # 検索結果のタイトル
                        'url': str,              # 結果のURL
                        'content': str,          # コンテンツの要約・抜粋
                        'score': float,          # 関連性スコア
                        'raw_content': str|None  # 完全なコンテンツ（あれば）
                    },
                    ...
                ]
            }
    """

    # APIリクエスト用のヘッダー設定
    # ヘッダーには認証トークンやコンテンツタイプなどの必須情報を含む
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"  # 環境変数からAPIキーを取得
    }
    
    # 結果を格納するリストを初期化
    search_docs = []
    
    # 各検索クエリに対して処理を実行
    for query in search_queries:
        # APIリクエストのペイロードを構築
        # モデルにsonar-proを指定し、システムとユーザーメッセージを設定
        payload = {
            "model": "sonar-pro",  # Perplexityの検索特化モデル
            "messages": [
                {
                    "role": "system",
                    "content": "ウェブを検索し、情報源とともに事実に基づいた情報を提供してください。"  # システム指示
                },
                {
                    "role": "user",
                    "content": query  # ユーザーからのクエリ
                }
            ]
        }
        
        # POSTリクエストを送信してレスポンスを取得
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",  # Perplexity APIのエンドポイント
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # エラーコードでレスポンスが返された場合、例外を発生
        
        # レスポンスJSONを解析
        data = response.json()
        content = data["choices"][0]["message"]["content"]  # LLMが生成した回答文
        
        # 引用情報（URL）を取得（なければデフォルトURLを使用）
        citations = data.get("citations", ["https://perplexity.ai"])
        
        # この検索クエリの結果リストを初期化
        results = []
        
        # 最初の引用が最も重要なソースと見なし、完全な内容を含める
        results.append({
            "title": f"Perplexity Search, Source 1",  # タイトル
            "url": citations[0],  # 最初の引用URL
            "content": content,  # LLMによる回答全文
            "raw_content": content,  # raw_contentも同じ内容を設定
            "score": 1.0  # 最大スコア（Tavilyのフォーマットに合わせる）
        })
        
        # 追加の引用があれば、セカンダリソースとして追加
        # これらは内容を重複させないため、簡易説明のみ
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity Search, Source {i}",  # ソース番号付きタイトル
                "url": citation,  # 引用URL
                "content": "See primary source for full content",  # 簡易説明
                "raw_content": None,  # 内容は主ソースにのみ含める
                "score": 0.5  # セカンダリソースなので低いスコア
            })
        
        # Tavily形式に合わせた応答構造を作成
        search_docs.append({
            "query": query,  # 元のクエリ
            "follow_up_questions": None,  # 使用しない
            "answer": None,  # 使用しない
            "images": [],  # 画像なし
            "results": results  # 検索結果
        })
    
    # 全クエリの検索結果を返却
    return search_docs

@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    """Exa APIを使用してウェブ検索を実行する非同期関数
    
    Exa APIは高度なニューラル検索エンジンで、通常の検索エンジンよりも
    意味理解に優れており、特に専門的な検索や詳細な情報取得に適しています。
    この関数はExaの機能を活用し、複数のクエリに対して並行して検索を実行します。
    
    特徴:
    1. ドメイン指定・除外機能（特定サイトに限定した検索が可能）
    2. サブページ取得機能（検索結果の関連ページも取得可能）
    3. コンテンツ長の制限設定（トークン数を抑制するため）
    4. AI生成の要約機能（検索結果の理解を助けるため）
    
    Args:
        search_queries (List[str]): 検索クエリの文字列リスト
        max_characters (int, optional): 各結果の生テキスト最大文字数
                                       Noneの場合はtextパラメータがTrueに設定される
        num_results (int): 検索結果の最大数（デフォルト: 5）
        include_domains (List[str], optional): 検索を限定するドメインリスト
                                              指定すると、これらのドメインからの結果のみ返される
        exclude_domains (List[str], optional): 検索から除外するドメインリスト
                                              include_domainsと同時に使用不可
        subpages (int, optional): 各検索結果から取得するサブページ数
                                 指定しない場合はサブページは取得されない
        
    Returns:
        List[dict]: 検索応答のリスト。各応答は他の検索APIと互換性のある形式
    """

    # include_domainsとexclude_domainsの両方が指定されていないか確認
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")
    
    # Exa APIクライアントの初期化
    # APIキーは環境変数から取得
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")
    
    # 単一クエリを処理する内部関数を定義
    async def process_query(query):
        # イベントループを取得して非同期処理を実行
        loop = asyncio.get_event_loop()
        
        # Exaの同期APIを非同期で実行するための関数
        def exa_search_fn():
            # 検索パラメータの辞書を構築
            kwargs = {
                # max_charactersがNoneの場合はtextをTrue、それ以外は文字数制限付きで設定
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # AI生成の要約を取得（Exaの強力な機能）
                "num_results": num_results  # 取得する結果数
            }
            
            # オプションパラメータを条件付きで追加
            if subpages is not None:
                kwargs["subpages"] = subpages
                
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
                
            # Exa APIで検索実行
            return exa.search_and_contents(query, **kwargs)
        
        # 同期関数を非同期実行
        response = await loop.run_in_executor(None, exa_search_fn)
        
        # 検索結果を整形
        formatted_results = []
        seen_urls = set()  # URL重複を避けるためのセット
        
        # オブジェクトからプロパティを安全に取得するヘルパー関数
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default
        
        # レスポンスから結果リストを取得
        results_list = get_value(response, 'results', [])
        
        # メイン検索結果を処理
        for result in results_list:
            # 結果のスコアを取得（デフォルト: 0.0）
            score = get_value(result, 'score', 0.0)
            
            # テキスト内容と要約を取得
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            
            # テキスト内容を設定（要約があれば追加）
            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content
            
            # タイトルとURLを取得
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            
            # 既に見たURLならスキップ（重複排除）
            if url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            # 結果エントリを作成
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            }
            
            # 整形済み結果リストに追加
            formatted_results.append(result_entry)
        
        # サブページが要求されていれば処理
        if subpages is not None:
            for result in results_list:
                # サブページリストを取得
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # サブページのスコアを取得
                    subpage_score = get_value(subpage, 'score', 0.0)
                    
                    # サブページのテキストと要約を取得・結合
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    
                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary
                    
                    subpage_url = get_value(subpage, 'url', '')
                    
                    # URLが既に見たものならスキップ
                    if subpage_url in seen_urls:
                        continue
                        
                    seen_urls.add(subpage_url)
                    
                    # サブページ結果を追加
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })
        
        # 画像があれば収集（メイン結果からのみ）
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # 重複画像を避ける
                images.append(image)
                
        # 検索応答を整形して返却
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results
        }
    
    # すべてのクエリを処理（レート制限を考慮して遅延を入れる）
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # 最初のリクエスト以外は短い遅延を入れる（レート制限対策）
            if i > 0:  
                await asyncio.sleep(0.25)
            
            # クエリを処理して結果を追加
            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # エラー処理（リクエスト失敗時でも処理を継続）
            print(f"Error processing query '{query}': {str(e)}")
            # エラー情報を含むプレースホルダー結果を追加
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
            
            # レート制限エラーの場合は待機時間を長くする
            if "429" in str(e):
                print("Rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(1.0)
    
    # すべての検索結果を返却
    return search_docs

@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """arXivから学術論文を検索する非同期関数
    
    arXiv（アーカイブ）は物理学、数学、コンピュータサイエンスなどの分野の
    プレプリント論文を公開している無料のリポジトリです。この関数は与えられた
    クエリに基づいてarXivから論文を検索し、その内容やメタデータを取得します。
    
    特徴:
    1. 論文全文の取得機能（必要に応じてPDFコンテンツをテキストに変換）
    2. 豊富なメタデータ取得（著者、カテゴリ、公開日など）
    3. arXivのレート制限に対応（3秒間隔で実行）
    4. 他の検索API関数と互換性のある結果形式
    
    Args:
        search_queries (List[str]): 検索クエリまたは論文IDのリスト
        load_max_docs (int, optional): クエリごとに取得する最大文書数（デフォルト: 5）
        get_full_documents (bool, optional): 全文を取得するかどうか（デフォルト: True）
        load_all_available_meta (bool, optional): 全メタデータを取得するか（デフォルト: True）

    Returns:
        List[dict]: 検索応答のリスト。各応答は他の検索APIと互換性のある形式
    """
    
    async def process_single_query(query):
        try:
            # 各クエリ用にarXivレトリーバーを作成
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            
            # 同期レトリーバーをスレッドプールで非同期実行
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            
            results = []
            # 結果の順序に基づいて減少するスコアを割り当て
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # メタデータを抽出
                metadata = doc.metadata
                
                # entry_idをURLとして使用（これが実際のarXivリンク）
                url = metadata.get('entry_id', '')
                
                # すべての有用なメタデータを含む内容を整形
                content_parts = []

                # 主要情報
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # 出版情報を追加
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # 追加メタデータを追加（存在する場合）
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # PDFリンクがあれば取得（リンクリストから）
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # すべてのコンテンツを改行で結合
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # entry_idをURLとして使用
                    'content': content,  # 整形されたメタデータ
                    'score': base_score - (i * score_decrement),  # 順序に基づくスコア
                    'raw_content': doc.page_content if get_full_documents else None  # 論文全文（要求された場合）
                }
                results.append(result)
                
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # 例外を適切にハンドリング
            print(f"Error processing arXiv query '{query}': {str(e)}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # arXivのレート制限（3秒ごとに1リクエスト）を尊重しながらクエリを処理
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # 最初のクエリ以外は遅延を追加（arXivのレート制限対策）
            if i > 0:
                await asyncio.sleep(3.0)
            
            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # 例外を適切にハンドリング
            print(f"Error processing arXiv query '{query}': {str(e)}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # レート制限エラーの場合は追加の遅延
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("ArXiv rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(5.0)  # レート制限エラーの場合は長めの遅延
    
    return search_docs

@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """PubMedから医学・生物学文献を検索する非同期関数
    
    PubMedは米国国立医学図書館（National Library of Medicine）が運営する
    世界最大の生物医学文献データベースです。この関数は与えられたクエリに基づいて
    PubMedから論文を検索し、その内容やメタデータを取得します。
    
    特徴:
    1. 医学・生物学分野の専門文献へのアクセス
    2. 論文メタデータの取得（発行日、著者、要約など）
    3. PubMedのAPI制限に適応的に対応（遅延を自動調整）
    4. 他の検索API関数と互換性のある結果形式を提供
    
    Args:
        search_queries (List[str]): 検索クエリのリスト
        top_k_results (int, optional): 各クエリで返す最大結果数（デフォルト: 5）
        email (str, optional): PubMed APIに必要なメールアドレス（NCBI要件）
        api_key (str, optional): 高いレート制限のためのPubMed APIキー
        doc_content_chars_max (int, optional): 文書内容の最大文字数（デフォルト: 4000）

    Returns:
        List[dict]: 検索応答のリスト。各応答は他の検索APIと互換性のある形式
    """
    
    async def process_single_query(query):
        try:
            # PubMed検索用のラッパーを作成
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,  # 返す結果の最大数
                doc_content_chars_max=doc_content_chars_max,  # 要約の最大文字数
                email=email if email else "your_email@example.com",  # NCBIが要求するメールアドレス
                api_key=api_key if api_key else ""  # オプションのAPIキー
            )
            
            # 同期的なラッパーをイベントループのスレッドプールで非同期実行
            loop = asyncio.get_event_loop()
            
            # lazy_loadメソッドで検索実行（より詳細な制御が可能）
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # 順序に基づく減少スコアを設定（最初の結果が最高スコア）
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # メタデータを整形してコンテンツを作成
                content_parts = []
                
                # 発行日情報
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                # 著作権情報
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                # 論文要約
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # 論文IDからPubMed URLを生成
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # すべてのコンテンツを改行で結合
                content = "\n".join(content_parts)
                
                # 結果エントリを作成
                result = {
                    'title': doc.get('Title', ''),  # 論文タイトル
                    'url': url,  # PubMedページURL
                    'content': content,  # 整形されたメタデータ
                    'score': base_score - (i * score_decrement),  # 順序ベースのスコア
                    'raw_content': doc.get('Summary', '')  # 論文要約を生コンテンツとして
                }
                results.append(result)
            
            # 検索結果を標準形式で返却
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # 詳細なエラー情報を提供
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # デバッグ用にスタックトレースを出力
            
            # エラー情報を含む空の結果を返却
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # すべてのクエリを適切な遅延を入れて処理
    search_docs = []
    
    # 適応的な遅延：小さく始めて必要に応じて増加
    delay = 1.0  # 初期遅延は控えめに設定
    
    for i, query in enumerate(search_queries):
        try:
            # 最初のリクエスト以外は遅延を入れる
            if i > 0:
                await asyncio.sleep(delay)
            
            # クエリを処理して結果を追加
            result = await process_single_query(query)
            search_docs.append(result)
            
            # 結果が得られた場合は遅延をやや減少（ただし最低0.5秒は維持）
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)
            
        except Exception as e:
            # メインループでのエラー処理
            error_msg = f"Error in main loop processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            
            # エラー情報を含む結果を追加
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # エラー発生時は遅延を増加（最大5秒まで）
            delay = min(5.0, delay * 1.5)
    
    # すべての検索結果を返却
    return search_docs

@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """Linkup APIを使用したウェブ検索を実行する非同期関数
    
    Linkup APIは新しいタイプの検索エンジンで、標準的な検索とより深い内容分析を
    組み合わせています。この関数は複数の検索クエリを並行して処理し、
    結果を標準化された形式で返します。
    
    特徴:
    1. 検索深度の調整機能（standard/deep）
    2. 非同期処理による効率的な複数クエリ処理
    3. 他の検索API関数と互換性のある出力形式
    
    Args:
        search_queries (List[str]): 検索クエリの文字列リスト
        depth (str, optional): 検索の深さ。"standard"（デフォルト）または"deep"
                              詳細はLinkupのドキュメント参照: https://docs.linkup.so/pages/documentation/get-started/concepts
        
    Returns:
        List[dict]: 検索応答のリスト。各応答には以下の形式の結果が含まれる:
            {
                'results': [            # 検索結果のリスト
                    {
                        'title': str,   # 検索結果のタイトル
                        'url': str,     # 結果のURL
                        'content': str, # コンテンツの要約
                    },
                    ...
                ]
            }
    """
    # 1. LinkupClientインスタンスを初期化
    # このクライアントは環境変数からAPIキーを自動的に取得
    client = LinkupClient()
    
    # 2. すべての検索クエリに対する非同期タスクを作成
    search_tasks = []
    for query in search_queries:
        # 各クエリに対して非同期検索を準備
        # - query: 検索したいテキスト
        # - depth: 検索の深さ（standard/deep）
        # - output_type: "searchResults"形式で結果を取得
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    # 3. すべての検索タスクを並行実行して結果を収集
    search_results = []
    # asyncio.gatherを使用して全てのタスクを並行実行
    for response in await asyncio.gather(*search_tasks):
        # 4. 各レスポンスを標準形式に変換
        search_results.append(
            {
                "results": [
                    # 各結果を辞書形式に変換
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    # 5. すべての検索結果を返却
    return search_results

@traceable
async def duckduckgo_search(search_queries):
    """
    DuckDuckGoを利用して複数の検索クエリを非同期で処理する関数
    
    引数:
        search_queries (List[str]): 処理する検索クエリのリスト
        
    戻り値:
        List[dict]: 検索結果のリスト
    """
    
    # 単一クエリを処理するための内部関数
    async def process_single_query(query):
        # イベントループを取得
        loop = asyncio.get_event_loop()
        
        # 同期的な検索処理を行う内部関数を定義
        def perform_search():
            results = []
            # DDGSクラスを使用してDuckDuckGo検索を実行
            with DDGS() as ddgs:
                # 最大5件の結果を取得
                ddg_results = list(ddgs.text(query, max_results=5))
                
                # 検索結果を整形
                for i, result in enumerate(ddg_results):
                    results.append({
                        'title': result.get('title', ''),  # 検索結果のタイトル（存在しない場合は空文字）
                        'url': result.get('link', ''),     # 検索結果のURL（存在しない場合は空文字）
                        'content': result.get('body', ''), # 検索結果の本文（存在しない場合は空文字）
                        'score': 1.0 - (i * 0.1),          # 順位に基づく簡易スコア付け（上位ほど高スコア）
                        'raw_content': result.get('body', '') # 加工していない本文
                    })
            # フォーマット済みの結果を返却
            return {
                'query': query,                 # 元の検索クエリ
                'follow_up_questions': None,    # フォローアップ質問（未実装）
                'answer': None,                 # 回答（未実装）
                'images': [],                   # 画像リスト（未実装）
                'results': results              # 検索結果のリスト
            }
        
        # 同期関数を非同期的に実行するためにイベントループのスレッドプールを使用
        return await loop.run_in_executor(None, perform_search)

    # すべてのクエリに対して並行処理を行うタスクを作成
    tasks = [process_single_query(query) for query in search_queries]
    # すべてのタスクを並行実行して結果を待機
    search_docs = await asyncio.gather(*tasks)
    
    return search_docs

@traceable
async def google_search_async(search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True):
    """Google検索を実行する非同期関数（API利用または代替スクレイピング）
    
    この関数は、Google検索を利用してウェブ検索を行います。環境変数にAPIキーと
    カスタム検索エンジンIDが設定されていれば公式APIを使用し、なければ代替手段として
    ウェブスクレイピングを使用します。複数のクエリを効率的に処理し、また必要に応じて
    検索結果のフルコンテンツも取得します。
    
    特徴:
    1. APIとスクレイピングの自動切り替え機能
    2. 並行リクエスト制限による安全な実行
    3. 適応的なエラー処理と待機戦略
    4. フルページコンテンツの取得オプション
    5. 他の検索API関数と互換性のある出力形式
    
    Args:
        search_queries (Union[str, List[str]]): 検索クエリの文字列または文字列リスト
        max_results (int): 各クエリで取得する最大結果数（デフォルト: 5）
        include_raw_content (bool): 検索結果の完全なページ内容を取得するか（デフォルト: True）
        
    Returns:
        List[dict]: 検索応答のリスト。各応答は他の検索API関数と互換性のある形式
    """
    # 1. 環境変数からAPIキーと検索エンジンIDを取得
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)  # 両方の値があれば公式APIを使用
    
    # 2. 検索クエリが単一の文字列の場合、リストに変換
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    # 3. ランダムなユーザーエージェント文字列を生成する関数
    def get_useragent():
        """ランダムなユーザーエージェント文字列を生成。検出回避に役立つ。"""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"
    
    # 4. 同期操作のためのエグゼキュータを作成（API使用時は不要）
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    # 5. 同時リクエスト数を制限するセマフォを作成
    semaphore = asyncio.Semaphore(5 if use_api else 2)  # APIとスクレイピングで異なる制限値
    
    # 6. 単一クエリを処理する内部関数を定義
    async def search_single_query(query):
        async with semaphore:  # セマフォを使用して同時リクエスト数を制限
            try:
                results = []
                
                # 6-A: APIベースの検索（API情報が利用可能な場合）
                if use_api:
                    # API結果は最大10件単位で取得するため、必要に応じて複数回リクエスト
                    for start_index in range(1, max_results + 1, 10):
                        # この回で取得する結果数を計算
                        num = min(10, max_results - (start_index - 1))
                        
                        # Google Custom Search APIへのリクエストパラメータを設定
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        print(f"Requesting {num} results for '{query}' from Google API...")

                        # リクエストを実行して結果を取得
                        async with aiohttp.ClientSession() as session:
                            async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API error: {response.status}, {error_text}")
                                    break
                                    
                                data = await response.json()
                                
                                # 検索結果を処理して標準形式に変換
                                for item in data.get('items', []):
                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": item.get('snippet', '')
                                    }
                                    results.append(result)
                        
                        # APIのクォータを尊重するため短い遅延を挿入
                        await asyncio.sleep(0.2)
                        
                        # 結果が期待数より少ない場合、追加リクエストは不要
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break
                
                # 6-B: スクレイピングベースの検索（API情報がない場合）
                else:
                    # ブロック検出を避けるためリクエスト間に短いランダム遅延を挿入
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"Scraping Google for '{query}'...")

                    # Googleをスクレイピングする内部関数を定義
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []
                            
                            # 要求された結果数に達するまで複数ページを取得
                            while fetched_results < max_results:
                                # Googleに検索リクエストを送信
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "*/*"
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies = {
                                        'CONSENT': 'PENDING+987',  # 同意ページをバイパス
                                        'SOCS': 'CAESHAgBEhIaAB',
                                    }
                                )
                                resp.raise_for_status()
                                
                                # BeautifulSoupで結果をパース
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0
                                
                                # 各検索結果を処理
                                for result in result_block:
                                    link_tag = result.find("a", href=True)
                                    title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                                    description_tag = result.find("span", class_="FrIlee")
                                    
                                    if link_tag and title_tag and description_tag:
                                        # GoogleのURLリダイレクトから実際のURLを抽出
                                        link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                                        
                                        # 重複URLをスキップ
                                        if link in fetched_links:
                                            continue
                                        
                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text
                                        
                                        # API結果と同じ形式で結果を格納
                                        search_results.append({
                                            "title": title,
                                            "url": link,
                                            "content": description,
                                            "score": None,
                                            "raw_content": description
                                        })
                                        
                                        fetched_results += 1
                                        new_results += 1
                                        
                                        # 最大結果数に達したら終了
                                        if fetched_results >= max_results:
                                            break
                                
                                # 新しい結果がなければ終了
                                if new_results == 0:
                                    break
                                    
                                # 次のページへ
                                start += 10
                                time.sleep(1)  # ページ間の遅延
                            
                            return search_results
                                
                        except Exception as e:
                            print(f"Error in Google search for '{query}': {str(e)}")
                            return []
                    
                    # スクレイピング関数をスレッドプールで非同期実行
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor, 
                        lambda: google_search(query, max_results)
                    )
                    
                    # 結果を格納
                    results = search_results
                
                # 7. 要求があれば各検索結果のフルコンテンツを取得
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)  # コンテンツ取得の同時実行数を制限
                    
                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []
                        
                        # 各結果のフルコンテンツを取得する内部関数
                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result['url']
                                headers = {
                                    'User-Agent': get_useragent(),
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                                }
                                
                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)  # ランダム遅延
                                    async with session.get(url, headers=headers, timeout=10) as response:
                                        if response.status == 200:
                                            # コンテンツタイプをチェックしてバイナリファイルを処理
                                            content_type = response.headers.get('Content-Type', '').lower()
                                            
                                            # PDFなどのバイナリファイルを処理
                                            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                                                result['raw_content'] = f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                            else:
                                                try:
                                                    # UTF-8でデコード（非UTF8文字は置換）
                                                    html = await response.text(errors='replace')
                                                    soup = BeautifulSoup(html, 'html.parser')
                                                    result['raw_content'] = soup.get_text()
                                                except UnicodeDecodeError as ude:
                                                    # デコードに問題があれば代替テキスト
                                                    result['raw_content'] = f"[Could not decode content: {str(ude)}]"
                                except Exception as e:
                                    print(f"Warning: Failed to fetch content for {url}: {str(e)}")
                                    result['raw_content'] = f"[Error fetching content: {str(e)}]"
                                return result
                        
                        # すべての結果のフルコンテンツ取得タスクを作成
                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))
                        
                        # すべてのタスクを並行実行
                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results
                        print(f"Fetched full content for {len(results)} results")
                
                # 8. 標準形式で結果を返却
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                print(f"Error in Google search for query '{query}': {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": []
                }
    
    try:
        # 9. すべての検索クエリに対するタスクを作成
        search_tasks = [search_single_query(query) for query in search_queries]
        
        # 10. すべての検索タスクを並行実行
        search_results = await asyncio.gather(*search_tasks)
        
        return search_results
    finally:
        # 11. エグゼキュータをクリーンアップ（作成されていた場合）
        if executor:
            executor.shutdown(wait=False)

async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """適切な検索APIを選択・実行し、結果を統一形式で返す関数
    
    この関数は「検索APIファサード」として機能し、複数の検索エンジンへの
    アクセスを統一的なインターフェースで提供します。指定された検索API名に
    応じて対応する検索関数を呼び出し、結果を標準化された形式で返します。
    
    主な役割:
    1. 検索API名に基づいて適切な検索処理関数を選択
    2. 検索パラメータを適切な形式で渡す
    3. 検索結果を統一された形式に整形
    4. 重複するコンテンツの除去と最大トークン数の制限
    
    Args:
        search_api (str): 使用する検索APIの名前（"tavily", "perplexity"など）
        query_list (list[str]): 実行する検索クエリのリスト
        params_to_pass (dict): 検索APIに渡す追加パラメータ
        
    Returns:
        str: 整形済みの検索結果を含む文字列
        
    Raises:
        ValueError: サポートされていない検索APIが指定された場合
    """
    # tavily検索API（一般的なウェブ検索向け）
    if search_api == "tavily":
        # 非同期で検索を実行（追加パラメータを辞書展開して渡す）
        search_results = await tavily_search_async(query_list, **params_to_pass)
        # 生コンテンツを含めずに結果を整形（容量削減のため）
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, include_raw_content=False)
    
    # perplexity検索API（AI生成要約付き検索）
    elif search_api == "perplexity":
        # Perplexityは同期APIのため、await不要
        search_results = perplexity_search(query_list, **params_to_pass)
        # 標準で生コンテンツを含める
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # exa検索API（ニューラル検索エンジン）
    elif search_api == "exa":
        # 非同期でニューラル検索を実行
        search_results = await exa_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # arxiv API（学術論文検索）
    elif search_api == "arxiv":
        # 物理学、数学、コンピュータサイエンスなどの学術論文を検索
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # pubmed API（医学・生物学文献検索）
    elif search_api == "pubmed":
        # 医学・生物学分野の専門文献データベースを検索
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # linkup検索API
    elif search_api == "linkup":
        # 深度調整可能な検索エンジン
        search_results = await linkup_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # DuckDuckGo検索（プライバシー重視の検索エンジン）
    elif search_api == "duckduckgo":
        # DuckDuckGoはパラメータをサポートしないため、params_to_passを使用しない
        search_results = await duckduckgo_search(query_list)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # Google検索（API使用またはスクレイピング）
    elif search_api == "googlesearch":
        # Google Custom Search APIまたはウェブスクレイピングによる検索
        search_results = await google_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    
    # サポートされていない検索API名の場合はエラー
    else:
        raise ValueError(f"Unsupported search API: {search_api}")
