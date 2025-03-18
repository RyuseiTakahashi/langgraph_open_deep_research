# Open Deep Research
 
Open Deep Researchはあらゆるトピックに関する調査を自動化し、カスタマイズ可能なレポートを作成するオープンソースアシスタントです。特定のモデル、プロンプト、レポート構造、検索ツールを使用して調査と執筆プロセスをカスタマイズすることができます。

![report-generation](https://github.com/user-attachments/assets/6595d5cd-c981-43ec-8e8b-209e4fefc596)

## 🚀 クイックスタート

希望する検索ツールとモデルのAPIキーが設定されていることを確認してください。

利用可能な検索ツール：

* [Tavily API](https://tavily.com/) - 一般的なウェブ検索
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - 一般的なウェブ検索
* [Exa API](https://exa.ai/) - ウェブコンテンツ用の強力なニューラル検索
* [ArXiv](https://arxiv.org/) - 物理学、数学、コンピュータサイエンスなどの学術論文
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - MEDLINE、ライフサイエンスジャーナル、オンラインブックからの生物医学文献
* [Linkup API](https://www.linkup.so/) - 一般的なウェブ検索
* [DuckDuckGo API](https://duckduckgo.com/) - 一般的なウェブ検索
* [Google Search API/Scrapper](https://google.com/) - カスタム検索エンジンを[こちら](https://programmablesearchengine.google.com/controlpanel/all)で作成し、APIキーを[こちら](https://developers.google.com/custom-search/v1/introduction)で取得

Open Deep Researchはレポート計画にプランナーLLM、レポート執筆にライターLLMを使用します：

* [`init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)と統合されている任意のモデルを選択できます
* サポートされる統合の完全なリストは[こちら](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)をご覧ください

### パッケージの使用方法

```bash
pip install open-deep-research
```

上記のように、LLMと検索ツールのAPIキーが設定されていることを確認してください：
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
```

Jupyterノートブックでの使用例については[src/open_deep_research/graph.ipynb](src/open_deep_research/graph.ipynb)をご覧ください：

グラフをコンパイルする：
```python
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

希望するトピックと設定でグラフを実行する：
```python
import uuid 
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "tavily",
                           "planner_provider": "anthropic",
                           "planner_model": "claude-3-7-sonnet-latest",
                           "writer_provider": "anthropic",
                           "writer_model": "claude-3-5-sonnet-latest",
                           "max_search_depth": 1,
                           }}

topic = "Overview of the AI inference market with focus on Fireworks, Together.ai, Groq"
async for event in graph.astream({"topic":topic,}, thread, stream_mode="updates"):
    print(event)
```

グラフはレポート計画が生成されると停止し、フィードバックを渡してレポート計画を更新できます：
```python
from langgraph.types import Command
async for event in graph.astream(Command(resume="Include a revenue estimate (ARR) in the sections"), thread, stream_mode="updates"):
    print(event)
```

レポート計画に満足したら、`True`を渡してレポート生成に進むことができます：
```python
async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
    print(event)
```

### LangGraph Studio UIをローカルで実行する

リポジトリをクローンする：
```bash
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research
```

`.env`ファイルにAPIキーを編集する（例えば、デフォルト選択用のAPIキーは以下の通り）：
```bash
cp .env.example .env
```

モデルと検索ツールに必要なAPIを設定します。

利用可能ないくつかのモデルとツール統合の例を以下に示します：
```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
export PERPLEXITY_API_KEY=<your_perplexity_api_key>
export EXA_API_KEY=<your_exa_api_key>
export PUBMED_API_KEY=<your_pubmed_api_key>
export PUBMED_EMAIL=<your_email@example.com>
export LINKUP_API_KEY=<your_linkup_api_key>
export GOOGLE_API_KEY=<your_google_api_key>
export GOOGLE_CX=<your_google_custom_search_engine_id>
```

LangGraphサーバーをローカルで起動し、ブラウザで開きます：

#### Mac

```bash
# uvパッケージマネージャーをインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係をインストールしてLangGraphサーバーを起動
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

#### Windows / Linux

```powershell
# 依存関係をインストール
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# LangGraphサーバーを起動
langgraph dev
```

Studio UIを開くには以下を使用します：
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

(1) `Topic`を提供して`Submit`をクリックします：

<img width="1326" alt="input" src="https://github.com/user-attachments/assets/de264b1b-8ea5-4090-8e72-e1ef1230262f" />

(2) これによりレポート計画が生成され、ユーザーによるレビューのために表示されます。

(3) フィードバック付きの文字列（`"..."`）を渡して、そのフィードバックに基づいて計画を再生成することができます。

<img width="1326" alt="feedback" src="https://github.com/user-attachments/assets/c308e888-4642-4c74-bc78-76576a2da919" />

(4) または、`true`を渡して計画を承認することもできます。

<img width="1480" alt="accept" src="https://github.com/user-attachments/assets/ddeeb33b-fdce-494f-af8b-bd2acc1cef06" />

(5) 承認されると、レポートのセクションが生成されます。

<img width="1326" alt="report_gen" src="https://github.com/user-attachments/assets/74ff01cc-e7ed-47b8-bd0c-4ef615253c46" />

レポートはマークダウン形式で生成されます。

<img width="1326" alt="report" src="https://github.com/user-attachments/assets/92d9f7b7-3aea-4025-be99-7fb0d4b47289" />

## 📖 レポートのカスタマイズ

いくつかのパラメータを通じて調査アシスタントの動作をカスタマイズできます：

- `report_structure`：レポートのカスタム構造を定義（デフォルトは標準的な研究レポート形式）
- `number_of_queries`：セクションごとに生成する検索クエリの数（デフォルト：2）
- `max_search_depth`：リフレクションと検索の反復の最大数（デフォルト：2）
- `planner_provider`：計画フェーズのモデルプロバイダ（デフォルト："anthropic"、ただし[こちら](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)にリストされている`init_chat_model`でサポートされる統合からの任意のプロバイダ可）
- `planner_model`：計画用の特定のモデル（デフォルト："claude-3-7-sonnet-latest"）
- `writer_provider`：執筆フェーズのモデルプロバイダ（デフォルト："anthropic"、ただし[こちら](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)にリストされている`init_chat_model`でサポートされる統合からの任意のプロバイダ可）
- `writer_model`：レポート執筆用のモデル（デフォルト："claude-3-5-sonnet-latest"）
- `search_api`：ウェブ検索に使用するAPI（デフォルト："tavily"、オプションには"perplexity"、"exa"、"arxiv"、"pubmed"、"linkup"を含む）

これらの設定により、調査の深さの調整から異なるレポート生成フェーズに特定のAIモデルを選択することまで、ニーズに基づいて調査プロセスを微調整できます。

### 検索API設定

すべての検索APIが追加の設定パラメータをサポートしているわけではありません。サポートしているものは以下の通りです：

- **Exa**: `max_characters`, `num_results`, `include_domains`, `exclude_domains`, `subpages`
  - 注意：`include_domains`と`exclude_domains`は一緒に使用できません
  - 特に特定の信頼できるソースに調査を絞り込む必要がある場合、情報の正確性を確保する場合、または調査が特定のドメイン（学術ジャーナル、政府サイトなど）の使用を必要とする場合に役立ちます
  - 特定のクエリに合わせたAI生成の要約を提供し、検索結果から関連情報を抽出しやすくします
- **ArXiv**: `load_max_docs`, `get_full_documents`, `load_all_available_meta`
- **PubMed**: `top_k_results`, `email`, `api_key`, `doc_content_chars_max`
- **Linkup**: `depth`

Exa設定の例：
```python
thread = {"configurable": {"thread_id": str(uuid.uuid4()),
                           "search_api": "exa",
                           "search_api_config": {
                               "num_results": 5,
                               "include_domains": ["nature.com", "sciencedirect.com"]
                           },
                           # その他の設定...
                           }}
```

### モデルの考慮事項

(1) [`init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/)と統合されている任意のプランナーモデルとライターモデルを渡すことができます。サポートされる統合の完全なリストは[こちら](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)をご覧ください。

(2) **プランナーモデルとライターモデルは構造化出力をサポートする必要があります**：使用するモデルが構造化出力をサポートしているかどうかは[こちら](https://python.langchain.com/docs/integrations/chat/)で確認してください。

(3) Groqでは、`on_demand`サービス階層を使用している場合、1分あたりのトークン（TPM）制限があります：
- `on_demand`サービス階層には`6000 TPM`の制限があります
- Groqモデルでセクションを書くには[有料プラン](https://github.com/cline/cline/issues/47#issuecomment-2640992272)が必要です

(4) `deepseek-R1`は[関数呼び出しが得意ではありません](https://api-docs.deepseek.com/guides/reasoning_model)。アシスタントはレポートセクションとレポートセクション評価のために構造化出力を生成するために関数呼び出しを使用します。例のトレースは[こちら](https://smith.langchain.com/public/07d53997-4a6d-4ea8-9a1f-064a85cd6072/r)をご覧ください。
- OpenAI、Anthropic、Groqの`llama-3.3-70b-versatile`のような特定のOSSモデルなど、関数呼び出しが得意なプロバイダを検討してください。
- 次のようなエラーが表示される場合、モデルが構造化出力を生成できないことが原因である可能性があります（[トレース](https://smith.langchain.com/public/8a6da065-3b8b-4a92-8df7-5468da336cbe/r)参照）：
```
groq.APIError: Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.
```

## 仕組み
   
1. `計画と実行` - Open Deep Researchは[計画と実行のワークフロー](https://github.com/assafelovic/gpt-researcher)に従い、計画を調査から分離して、より時間のかかる調査フェーズの前にレポート計画のヒューマンインザループの承認を可能にします。デフォルトでは、レポートセクションを計画するために[推論モデル](https://www.youtube.com/watch?v=f0RbwrBcFmc)を使用します。この段階では、レポートトピックに関する一般的な情報を収集するためにウェブ検索を使用し、レポートセクションの計画を支援します。しかし、レポートセクションを導くためにユーザーからレポート構造を受け入れるとともに、レポート計画に関する人間のフィードバックも受け入れます。
   
2. `調査と執筆` - レポートの各セクションは並行して書かれます。調査アシスタントは[Tavily API](https://tavily.com/)、[Perplexity](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api)、[Exa](https://exa.ai/)、[ArXiv](https://arxiv.org/)、[PubMed](https://pubmed.ncbi.nlm.nih.gov/)または[Linkup](https://www.linkup.so/)を通じてウェブ検索を使用して、各セクションのトピックに関する情報を収集します。各レポートセクションについて振り返り、ウェブ検索のためのフォローアップ質問を提案します。この調査の「深さ」はユーザーが望む反復回数だけ進みます。紹介や結論などの最終セクションは、レポートの本文が書かれた後に書かれ、レポートが一貫性と整合性を確保するのに役立ちます。プランナーは計画段階で本文セクションと最終セクションを決定します。

3. `異なるタイプの管理` - Open Deep ResearchはLangGraphをベースに構築されており、[アシスタントを使用](https://langchain-ai.github.io/langgraph/concepts/assistants/)した設定管理をネイティブにサポートしています。レポートの`構造`はグラフ設定のフィールドであり、ユーザーが異なるタイプのレポート用に異なるアシスタントを作成できるようにします。

## UX

### ローカルデプロイメント

[クイックスタート](#-クイックスタート)に従ってLangGraphサーバーをローカルで起動します。

### ホステッドデプロイメント
 
[LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)に簡単にデプロイできます。

