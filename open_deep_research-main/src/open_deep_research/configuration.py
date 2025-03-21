# -----------------------------------------------------------------------
# configuration.py - Open Deep Researchのグローバル設定管理
# -----------------------------------------------------------------------
# このファイルは、レポート生成プロセスの全体的な動作を制御する設定を定義します。
# 検索API選択、モデル選択、レポート構造など、システムの中核設定を管理します。
# -----------------------------------------------------------------------

import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

# レポートのデフォルト構造を定義
# これは、特に指定がない場合に使用される標準的なレポートテンプレートです
# 導入、本文セクション、結論の3部構成から成ります
DEFAULT_REPORT_STRUCTURE = """
ユーザーが提供したトピックに関するレポートを作成するには、この構造を使用してください：

1. 導入部（調査不要）
    - トピック領域の簡潔な概要

2. 本文セクション：
    - 各セクションはユーザーが提供したトピックのサブトピックに焦点を当てる
    
3. 結論
    - 本文セクションを要約する構造的要素（リストまたは表）を1つ含める 
    - レポートの簡潔なまとめを提供する
"""

# 利用可能な検索APIを列挙型で定義
# 各検索APIは異なる特性を持ち、検索ターゲットや必要な詳細度に応じて選択できます
class SearchAPI(Enum):
    # 一般的なWeb検索と要約機能を持つAPI
    PERPLEXITY = "perplexity"
    # 一般的なウェブ検索向けのAPI
    TAVILY = "tavily"
    # ニューラル検索機能付きの高度なウェブ検索API
    EXA = "exa"
    # 物理学、数学、コンピュータサイエンスなどの学術論文専用API
    ARXIV = "arxiv"
    # 医学および生物医学文献専用の検索API
    PUBMED = "pubmed"
    # 一般的なウェブ検索用のAPI
    LINKUP = "linkup"
    # プライバシー重視の一般検索エンジンAPI
    DUCKDUCKGO = "duckduckgo"
    # Googleの検索機能にアクセスするAPI（要APIキーとカスタム検索エンジンID）
    GOOGLESEARCH = "googlesearch"

# アプリケーション全体の設定を管理するデータクラス
# LangGraphフレームワークと統合され、実行時設定を提供します
@dataclass(kw_only=True)
class Configuration:
    """レポート生成システムの設定パラメータを管理するクラス。
    
    すべての設定はデフォルト値を持ち、実行時に上書き可能です。
    設定は環境変数またはRunnable設定から読み込まれます。
    """
    
    # レポートの全体構造を定義（カスタマイズ可能）
    report_structure: str = DEFAULT_REPORT_STRUCTURE  # レポートの構造テンプレート
    
    # 検索パラメータ
    number_of_queries: int = 2  # 各イテレーションで生成する検索クエリの数
    max_search_depth: int = 2  # 最大検索深度（反復的な検索+リフレクションのサイクル数）
    
    # レポート計画用のAIモデル設定
    planner_provider: str = "anthropic"  # 計画立案に使用するAIプロバイダー
    planner_model: str = "claude-3-7-sonnet-latest"  # 計画立案に使用するモデル（デフォルトはClaude 3.7 Sonnet）
    
    # レポート執筆用のAIモデル設定
    writer_provider: str = "anthropic"  # レポート執筆に使用するAIプロバイダー
    writer_model: str = "claude-3-5-sonnet-latest"  # レポート執筆に使用するモデル（デフォルトはClaude 3.5 Sonnet）
    
    # 検索API設定
    search_api: SearchAPI = SearchAPI.TAVILY  # 使用する検索API（デフォルトはTavily）
    search_api_config: Optional[Dict[str, Any]] = None  # 検索API固有の追加設定（オプション）

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """LangGraphのRunnableConfigから設定インスタンスを作成します。
        
        この方法により、実行時に柔軟な設定が可能になります。設定値の優先順位は：
        1. 提供されたRunnable設定の値
        2. 環境変数（大文字で設定名が一致するもの）
        3. このクラスで定義されたデフォルト値
        
        Args:
            config: LangGraphから提供されるRunnableConfig（オプション）
            
        Returns:
            設定済みのConfigurationインスタンス
        """
        # 'configurable'キーがあれば抽出、なければ空辞書
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # 設定値を収集：環境変数 > configurable辞書の値 > デフォルト値（優先順位）
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init  # 初期化可能なフィールドのみ処理
        }
        
        # 値が存在する項目のみでConfigurationインスタンスを作成
        return cls(**{k: v for k, v in values.items() if v})