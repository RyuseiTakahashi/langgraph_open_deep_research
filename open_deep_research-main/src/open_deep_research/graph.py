from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from open_deep_research.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback
)

from open_deep_research.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from open_deep_research.configuration import Configuration
from open_deep_research.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

## Nodes -- 
async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """トピックに基づいたレポート計画を生成する関数
    
    この関数はレポート生成プロセス全体の起点となる重要なステップです。ユーザーが提供した
    トピックと設定情報を基に、AIを活用して構造化されたレポート計画を作成します。
    
    処理の詳細フロー:
    1. 入力データの取得:
       - ユーザーが提供したトピック（例: "AIの進化と応用"）
       - 以前のフィードバック（反復処理の場合のみ存在）
       
    2. 設定情報の抽出と準備:
       - レポート構造テンプレート（デフォルトまたはカスタム）
       - 検索パラメータ（クエリ数、検索API種別など）
       - 検索API固有の設定（例: Exaのドメインフィルタ）
       
    3. 検索クエリ生成プロセス:
       - クエリ生成用のAIモデルを初期化（例: Claude 3.5 Sonnet）
       - 構造化出力を得るためのラッパー設定
       - トピックに最適な検索クエリを生成（例: "AIの歴史的発展", "AIの産業応用事例"）
       
    4. ウェブ検索の実行:
       - 生成されたクエリを使用して選択された検索API（Tavily, Perplexityなど）でウェブ検索
       - 複数クエリを非同期で処理して効率化
       - 検索結果を統合・重複排除して構造化テキストに変換
       
    5. レポート計画生成:
       - 計画立案用のAIモデルを初期化（例: Claude 3.7 Sonnet）
       - 検索結果とユーザーフィードバックを考慮したプロンプト構築
       - 「思考」モードを活用した深い分析（Claude 3.7 Sonnetの場合）
       - セクション構造の生成（導入、本文セクション、結論を含む）
    
    技術的詳細:
    - 非同期処理（async/await）により複数の検索クエリを効率的に処理
    - LangChainのinit_chat_modelを使用して異なるプロバイダのモデルに対応
    - 構造化出力（Sections型）により厳密な形式でレポート計画を取得
    - Claude 3.7 Sonnetには特別な「思考」設定を適用（適切なモデルが指定された場合）
    
    Args:
        state (ReportState): レポート生成プロセスの現在状態を格納するオブジェクト
            - topic: レポートのメインテーマ (例: "生成AIの最新動向")
            - feedback_on_report_plan: 前回の計画に対するユーザーフィードバック（存在する場合）
            
        config (RunnableConfig): 実行時の設定情報とパラメータを含むオブジェクト
            - configurable: カスタム設定を含む辞書（プロバイダ、モデル名、検索API等）
    
    Returns:
        Dict[str, List[Section]]: 生成されたレポート計画を含む辞書
            - "sections": セクションオブジェクトのリスト。各セクションには以下が含まれる:
                - name: セクション名（例: "はじめに", "AIの産業応用"）
                - description: セクションの内容説明
                - research: 調査が必要かどうかの真偽値
                - content: 初期状態では空文字列
    
    例:
    返値の例: {"sections": [
        Section(name="はじめに", description="レポートの目的と範囲", research=False, content=""),
        Section(name="AI技術の現状", description="最新のAIアルゴリズムと手法", research=True, content=""),
        ...
    ]}
    """

    # 1. 入力データの取得
    topic = state["topic"]
    # ユーザーからのフィードバック（あれば）
    feedback = state.get("feedback_on_report_plan", None)

    # 2. 設定情報の抽出と準備
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # 検索API固有の設定
    params_to_pass = get_search_params(search_api, search_api_config)  # 検索用のパラメータを準備

    # JSONオブジェクトが渡された場合は文字列に変換
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # 3. 検索クエリ生成用のモデル設定
    writer_provider = get_config_value(configurable.writer_provider)  # モデルプロバイダー(OpenAI, Anthropicなど)
    writer_model_name = get_config_value(configurable.writer_model)  # モデル名(GPT-4, Claudeなど)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider)  # モデルの初期化
    structured_llm = writer_model.with_structured_output(Queries)  # 構造化出力のためのラッパー

    # 4. システムプロンプトの準備（検索クエリ生成用）
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        number_of_queries=number_of_queries
    )

    # 5. 検索クエリの生成
    results = structured_llm.invoke([
        SystemMessage(content=system_instructions_query),
        HumanMessage(content="レポートのセクション計画に役立つ検索クエリを生成してください。")
    ])

    # 6. ウェブ検索の実行
    query_list = [query.search_query for query in results.queries]  # クエリリストの抽出
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)  # 検索の実行

    # 7. レポート計画生成用のシステムプロンプト準備
    system_instructions_sections = report_planner_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        context=source_str, 
        feedback=feedback
    )

    # 8. レポート計画生成用のモデル設定
    planner_provider = get_config_value(configurable.planner_provider)  # 計画立案者モデルのプロバイダー
    planner_model = get_config_value(configurable.planner_model)  # 計画立案者モデル名

    # 9. レポート計画生成用のメッセージ準備
    planner_message = """
    レポートのセクションを生成してください。レスポンスには'sections'フィールドを含め、セクションのリストを記載する必要があります。
    各セクションには、name（名前）、description（説明）、plan（計画）、research（研究が必要かどうか）、content（内容）のフィールドが必要です。
    """

    # 10. 計画立案者モデルの初期化（モデルによって異なる設定）
    if planner_model == "claude-3-7-sonnet-latest":
        # Claude 3.7 Sonnetの場合は「思考」予算を設定（より深い分析のため）
        planner_llm = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider, 
            max_tokens=20_000, 
            thinking={"type": "enabled", "budget_tokens": 16_000}
        )
    else:
        # その他のモデルは標準設定で初期化
        planner_llm = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider
        )
    
    # 11. レポートセクションの生成
    structured_llm = planner_llm.with_structured_output(Sections)  # 構造化出力の設定
    report_sections = structured_llm.invoke([
        SystemMessage(content=system_instructions_sections),
        HumanMessage(content=planner_message)
    ])

    # 12. 生成されたセクションの取得と返却
    sections = report_sections.sections
    return {"sections": sections}

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """レポート計画に対するユーザーフィードバックを処理し、次のステップを決定する関数
    
    この関数はLangGraphワークフロー内の「人間とAIの協調」(Human-in-the-loop)を実現する
    重要なノードです。レポート生成プロセスの最初の分岐点として機能し、ユーザーが
    生成された計画を確認・修正できる機会を提供します。
    
    詳細な処理フロー:
    1. レポート計画の表示準備:
       - 状態から現在のレポートトピックとセクション情報を取得
       - セクション情報（名前、説明、調査必要性）を読みやすい形式に整形
       - 例: "Section: AIの歴史\nDescription: 1950年代から現在までのAI技術の進化\nResearch needed: Yes"
       
    2. ユーザーインタラクション:
       - LangGraphの`interrupt`関数を使用してワークフローを一時停止
       - ユーザーインターフェイスに整形されたレポート計画とフィードバック要求を表示
       - ユーザーの入力を待機（これにより同期的な処理が一時停止する）
       
    3. フィードバック解析と処理分岐:
       - 3-A: 承認パス（feedback = True）:
         * ユーザーが「true」を入力して計画を承認した場合
         * 調査が必要なセクション(research=True)のみを抽出
         * 各セクションごとに並行処理指示(Send命令)を生成
         * 例: 3つのセクションがあれば、3つの並行タスクが作成される
       
       - 3-B: 修正パス（feedback = 文字列）:
         * ユーザーが修正フィードバックを提供した場合
         * フィードバック内容を状態に保存
         * generate_report_planノードに処理を戻し、計画を再生成
         * 例: "第3セクションを分割し、AIの倫理的問題を追加してください"
       
       - 3-C: エラー処理（想定外のフィードバック型）:
         * boolでも文字列でもない値が返された場合にエラー処理
    
    技術的詳細:
    - `interrupt`関数: LangGraphの特殊関数で、ワークフローを一時停止しユーザー入力を待機する
    - `Command`オブジェクト: ワークフロー制御命令を表すLangGraphの構造体
      * goto: 次に実行するノード名またはノード名のリスト
      * update: 状態に適用する更新情報
    - `Send`命令: 並列実行を指示するLangGraphの特殊命令
      * 第1引数: 実行するノード名
      * 第2引数: 渡すデータ（新しい状態の一部）
    
    Args:
        state (ReportState): 現在のグラフ状態を表すオブジェクト
            - topic: レポートのメインテーマ (例: "量子コンピューティングの進展")
            - sections: Section型オブジェクトのリスト（計画されたレポート構造）
            
        config (RunnableConfig): ワークフロー設定情報（この関数では主に型定義のために存在）
        
    Returns:
        Command: 次のワークフロー実行ステップを指定するコマンドオブジェクト
            - 承認時: 複数のSend命令を含むgotoリストを持つCommand
            - 修正時: generate_report_planへのgotoと状態更新を持つCommand
    
    例:
    承認時の返値例: Command(goto=[Send("build_section_with_web_research", {...}), Send(...), ...])
    修正時の返値例: Command(goto="generate_report_plan", update={"feedback_on_report_plan": "セクション構成を変更して..."})
    """

    # 1. 現在のレポート計画をユーザー表示用に整形
    topic = state["topic"]  # レポートのメインテーマ
    sections = state['sections']  # 生成されたセクションリスト
    
    # セクションを読みやすい形式に変換
    # 各セクションの情報（名前、説明、調査必要性）を明確な形式で整形
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"  # セクション名
        f"Description: {section.description}\n"  # セクションの説明
        f"Research needed: {'Yes' if section.research else 'No'}\n"  # 調査が必要かどうか
        for section in sections
    )

    # 2. ユーザーにレポート計画を提示し、フィードバックを要求
    # interrupt関数はワークフローを一時停止し、UIでユーザーにメッセージを表示
    # このステップでプロセスは一時停止し、ユーザーの応答を待つ
    interrupt_message = f"""以下のレポート計画についてフィードバックをお願いします。
                        \n\n{sections_str}\n
                        \nこのレポート計画はあなたのニーズを満たしていますか？\n計画を承認する場合は「true」を入力してください。\nまたは、レポート計画を再生成するためのフィードバックを提供してください："""
    
    # フィードバックを待機（ユーザーが応答するまでここで処理が止まる）
    # interrupt関数はLangGraphの特殊機能で、UI経由でユーザーと対話できる
    feedback = interrupt(interrupt_message)

    # 3. ユーザーのフィードバックに基づいて処理を分岐
    # フィードバックの型と内容に基づいて、次の処理ステップを決定
    
    # 3-A: ユーザーが計画を承認した場合（True が返された）
    if isinstance(feedback, bool) and feedback is True:
        # 調査が必要なセクションそれぞれについて、並行して処理を開始
        # Command(goto=[...])構造で複数の並行ノード実行を指示
        return Command(goto=[
            # 各セクションに対してSend命令を作成（並列処理の指示）
            # Send命令は「このノードをこのデータで実行せよ」という指示
            Send("build_section_with_web_research", {
                "topic": topic,  # レポートのトピック
                "section": s,    # 処理対象のセクション
                "search_iterations": 0  # 検索反復回数の初期値（初回は0から開始）
            }) 
            # リスト内包表記で条件付きSend命令リストを生成
            # sections内の各要素sに対して反復するが、research=Trueのセクションのみ処理
            # これにより「はじめに」や「結論」などresearch=Falseのセクションは除外される
            for s in sections 
            if s.research
        ])
    
    # 3-B: ユーザーが修正フィードバックを提供した場合（文字列が返された）
    elif isinstance(feedback, str):
        # フィードバックを状態に保存して計画生成ノードに戻る
        # Command(goto="...", update={...})構造で状態更新と遷移先を指定
        return Command(
            goto="generate_report_plan",  # 計画生成ノードに移動（計画を再作成）
            update={"feedback_on_report_plan": feedback}  # 状態にフィードバックを追加
        )
    
    # 3-C: 想定外の型のフィードバックが返された場合（エラー処理）
    # UIバグやプログラムエラーでここに到達する可能性がある
    else:
        # 明示的なエラーを発生させてワークフローを停止
        # 型情報を含めることでデバッグを容易にする
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
def generate_queries(state: SectionState, config: RunnableConfig):
    """特定のセクションの調査用検索クエリを生成するノード関数
    
    この関数は調査プロセスの最初のステップとして、与えられたセクションに関する
    効果的な検索クエリを生成します。良質なクエリはレポートの品質に直接影響するため、
    このステップは全体のワークフローにおいて極めて重要です。
    
    詳細な処理フロー:
    1. 状態情報の取得:
       - レポート全体のトピック（例: "気候変動の経済的影響"）
       - 対象セクション情報（名前、説明など）
       - 例: セクション「農業への影響」の説明「気候変動が世界の農業生産性に与える影響」
       
    2. 設定情報の取得:
       - LangGraph設定から検索クエリ数を取得（デフォルト: 2）
       - Configuration.from_runnable_configを使用して設定を正規化
       - 例: number_of_queries = 3 （3つのクエリを生成）
       
    3. クエリ生成モデル準備:
       - 設定されたプロバイダ（OpenAI、Anthropicなど）とモデル名を取得
       - init_chat_modelを使用してLLMモデルを初期化
       - 構造化出力（Queries型）を返すようにモデルをラップ
       - 例: Claude 3.5 Sonnetモデルを準備し、構造化出力を設定
       
    4. プロンプト生成:
       - query_writer_instructionsテンプレートにパラメータを注入
       - トピック、セクショントピック、クエリ数を指定したシステムプロンプトを作成
       - プロンプト例: "農業への影響に関する3つの効果的な検索クエリを生成してください"
       
    5. クエリ生成実行:
       - 準備したモデルにシステムプロンプトと人間メッセージを送信
       - 構造化された検索クエリリストを取得
       - 例: ["気候変動 農業生産性 統計", "気温上昇 作物収量 変化", "干ばつ 農業経済 影響"]
    
    技術的詳細:
    - Queries型: 検索クエリのリストを保持する構造化データモデル（state.pyで定義）
    - LangChainの初期化プロセスを使用してモデルに依存せず動作
    - get_config_valueヘルパー関数でEnum値と文字列値の両方に対応
    
    Args:
        state (SectionState): セクション調査に関する現在の状態
            - topic: レポートの全体テーマ（例: "再生可能エネルギーの未来"）
            - section: 対象セクションの情報（Section型オブジェクト）
                - name: セクション名（例: "太陽光発電の進展"）
                - description: セクションの詳細説明
                - research: 調査フラグ（この関数が呼ばれる場合は常にTrue）
                
        config (RunnableConfig): LangGraphの実行時設定情報
            - configurable: カスタム設定情報（writer_provider, number_of_queriesなど）
        
    Returns:
        Dict[str, List[SearchQuery]]: 生成された検索クエリのリスト
            - "search_queries": SearchQuery型のリスト。各クエリは以下を含む:
                - search_query: 検索エンジンに渡す実際のクエリ文字列
    
    例:
    入力: state={"topic": "量子コンピューティング", "section": Section(name="商業応用", description="量子コンピュータの商業的利用例")}
    返値: {"search_queries": [SearchQuery(search_query="量子コンピューティング 商業応用 事例"), SearchQuery(search_query="量子アルゴリズム ビジネス 実装例")]}
    """

    # 1. 状態情報の取得
    # レポート全体のトピックとターゲットセクションの情報を状態から抽出
    topic = state["topic"]  # レポート全体のメインテーマ
    section = state["section"]  # 処理対象の特定セクション情報

    # 2. 設定情報の取得
    # LangGraphの設定からConfiguration構造体を作成し、クエリ数の設定値を取得
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries  # 生成するクエリの数（デフォルト: 2）

    # 3. クエリ生成モデル準備
    # 設定されたプロバイダ・モデルでLLMを初期化（例: Anthropic/Claude、OpenAI/GPT-4など）
    writer_provider = get_config_value(configurable.writer_provider)  # モデルプロバイダ（anthropic, openaiなど）
    writer_model_name = get_config_value(configurable.writer_model)   # モデル名（claude-3-5-sonnet-latestなど）
    # LangChainのinit_chat_modelを使用して統一的なインターフェースでモデルを初期化
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    # 構造化出力を使用してLLMの出力をQueries型に強制
    structured_llm = writer_model.with_structured_output(Queries)

    # 4. プロンプト生成
    # query_writer_instructionsテンプレートに必要な情報を注入してプロンプトを生成
    system_instructions = query_writer_instructions.format(
        topic=topic,                     # レポート全体のトピック
        section_topic=section.description, # 対象セクションの詳細説明  
        number_of_queries=number_of_queries  # 生成するクエリの数
    )

    # 5. クエリ生成実行
    # モデルを呼び出して検索クエリを生成
    # システムメッセージとシンプルな人間メッセージを使用する2段階のプロンプト構造
    queries = structured_llm.invoke([
        SystemMessage(content=system_instructions),  # 詳細なシステム指示（フォーマット済み）
        HumanMessage(content="Generate search queries on the provided topic.")  # シンプルな指示
    ])

    # 生成されたクエリのリストを含む辞書を返す
    # この値はsearch_webノードで次に使用される
    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """セクションのクエリに対してウェブ検索を実行する非同期ノード関数
    
    この関数はレポート作成の情報収集フェーズの中核であり、generate_queriesで生成された
    検索クエリを実行し、生のウェブコンテンツを取得します。複数の検索APIに対応し、
    非同期処理を活用することで効率的なデータ収集を実現します。
    
    詳細な処理フロー:
    1. クエリ情報の取得:
       - 状態から生成済みの検索クエリリストを取得
       - 例: ["量子コンピューティング 商業応用 最新事例", "量子ビット 実用化 進展"]
       
    2. 検索API設定の取得と準備:
       - 使用する検索API（Tavily, Perplexity, Exa, ArXiv, PubMedなど）を特定
       - API固有の設定パラメータを取得（結果数、ドメイン制限など）
       - パラメータをAPI仕様に合わせてフィルタリング
       - 例: Exaで特定ドメイン（nature.com, sciencedirect.com）の検索のみ行うよう設定
       
    3. 検索クエリのフォーマット:
       - SearchQuery型のリストから純粋な文字列クエリのリストを抽出
       - 例: ["クエリ1", "クエリ2", ...]
       
    4. ウェブ検索の実行:
       - select_and_execute_search関数を使用して適切な検索APIで検索を実行
       - 非同期処理（await）で効率的にリクエストを処理
       - この間バックグラウンドでは:
         * 選択されたAPIでリクエストを送信
         * 結果を取得して構造化
         * 重複コンテンツを除去
         * 文字列形式に整形
       
    5. 結果の返却:
       - 検索結果文字列と更新された検索反復カウントを返却
       - 反復カウントはwrite_section関数で品質評価に使用される
    
    技術的詳細:
    - async/await: 非同期I/O処理により複数のネットワークリクエストを効率的に処理
    - Configuration.from_runnable_config: LangGraph設定を統一的に扱うためのユーティリティ
    - select_and_execute_search: 複数の検索API実装をファサードパターンで隠蔽する関数
    - get_search_params: 検索API固有のパラメータのみを抽出するヘルパー関数
    
    Args:
        state (SectionState): セクション検索に関する現在の状態
            - search_queries: SearchQuery型オブジェクトのリスト（前ステップで生成）
            - search_iterations: 現在の検索反復回数（繰り返し調査の場合に使用）
            
        config (RunnableConfig): LangGraphの実行時設定情報
            - configurable: カスタム設定情報（search_api, search_api_configなど）
        
    Returns:
        Dict: 検索結果と更新された反復回数を含む辞書
            - "source_str": 整形された検索結果テキスト。セクション執筆の入力として使用
            - "search_iterations": インクリメントされた検索反復カウント
    
    例:
    入力: state={"search_queries": [SearchQuery(search_query="量子コンピューティング 商業応用"), ...], "search_iterations": 0}
    
    返値: {
        "source_str": "Content from sources:\n==========\nSource: Quantum Computing in Business\nURL: https://example.com/...\n===\nMost relevant content from source: 量子コンピューティングは金融業界で...\n===\nFull source content: ...\n==========\n...",
        "search_iterations": 1
    }
    """

    # 1. 検索クエリの取得
    # 状態から前ステップ（generate_queries）で生成された検索クエリリストを取得
    search_queries = state["search_queries"]  # SearchQuery型オブジェクトのリスト

    # 2. 検索API設定の取得
    # LangGraphの設定から、使用する検索APIと関連設定を抽出
    configurable = Configuration.from_runnable_config(config)  # 設定情報をConfiguration型に変換
    search_api = get_config_value(configurable.search_api)  # 使用する検索API（"tavily", "perplexity"など）
    search_api_config = configurable.search_api_config or {}  # API固有の設定（デフォルトは空辞書）
    
    # API固有のパラメータを抽出（APIが対応していないパラメータは除外）
    # 例：Exaには"include_domains"を渡すが、Tavilyには渡さない
    params_to_pass = get_search_params(search_api, search_api_config)

    # 3. 検索クエリのフォーマット
    # SearchQuery型オブジェクトから純粋な文字列のリストを抽出
    query_list = [query.search_query for query in search_queries]  # 文字列クエリのリスト作成

    # 4. ウェブ検索の実行
    # select_and_execute_search関数は、指定されたAPIに応じて適切な検索処理を選択実行
    # awaitキーワードにより、この処理は非同期で実行される（I/O待ちの間、他の処理を実行可能）
    source_str = await select_and_execute_search(
        search_api,      # 使用する検索API
        query_list,      # 検索クエリリスト
        params_to_pass   # API固有のパラメータ
    )

    # 5. 結果の返却
    # 検索結果と更新されたカウンターを含む辞書を返却
    # この返値はwrite_sectionノードで使用され、セクション内容の執筆に利用される
    return {
        "source_str": source_str,                        # 整形された検索結果テキスト
        "search_iterations": state["search_iterations"] + 1  # 検索反復カウントをインクリメント
    }

def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """レポートのセクションを執筆し、品質評価によって追加調査の必要性を判断する関数
    
    この関数はレポート生成プロセスの核心部分であり、3つの重要な役割を担います：
    1. 検索結果を基にしたセクション内容の執筆
    2. 執筆されたコンテンツの品質評価
    3. 評価結果に基づくワークフロー制御（完了または追加調査）
    
    このフィードバックループにより、単なる情報集約ではなく、高品質で整合性のある
    レポートセクションの生成を実現しています。
    
    詳細な処理フロー:
    1. 入力データの取得と準備:
       - レポートのトピック（例: "人工知能の倫理的側面"）
       - 対象セクション情報（名前、説明、既存内容など）
       - 検索結果テキスト（前ステップで取得したウェブコンテンツ）
       - 例: セクション「プライバシーへの影響」の検索結果
       
    2. セクション執筆準備:
       - 執筆用プロンプトの生成（トピック、セクション名、説明、検索結果を含む）
       - AIモデルの初期化と設定
       - 例: Claude 3.5 Sonnetを使ってセクション執筆準備
       
    3. セクション執筆:
       - AIモデルによるセクション内容の生成
       - 検索結果を統合した信頼性の高いコンテンツ作成
       - 既存コンテンツがある場合は、それを基に拡張・改善
       - 例: "AIによるデータ収集がプライバシーに与える影響について..."
       
    4. 品質評価準備:
       - 評価用プロンプトの生成（セクションの品質基準を含む）
       - 評価モデルの初期化（プランナーモデルを使用）
       - Claudeの「思考」機能を活用した深い分析（可能な場合）
       - 例: "このセクションは必要な情報を十分にカバーしているか評価する"
       
    5. 品質評価実行:
       - 執筆されたセクションの品質を評価
       - 不足情報や追加すべき観点の特定
       - 構造化出力（Feedback型）で評価結果を取得
       - 例: 評価="fail", フォローアップクエリ=["AIプライバシー規制 最新動向", ...]
       
    6. 評価に基づく分岐処理:
       - 6-A: 合格またはmax_search_depthに到達した場合:
         * セクションを完了済みとしてマーク
         * プロセスを終了してレポート統合フェーズへ進む
         
       - 6-B: 不合格で追加調査が必要な場合:
         * 評価モデルから提案された新しい検索クエリをセット
         * 検索プロセス（search_web）に戻って反復処理を継続
    
    技術的詳細:
    - Command型: LangGraphのワークフロー制御用の特殊型
      * goto: 次のノード名またはEND定数を指定
      * update: 状態の更新内容を指定
    
    - Feedbackループ: 「執筆→評価→改善」のサイクルを自動化
      * 適切な情報が得られるまで検索を繰り返す自己修正メカニズム
      * max_search_depthで無限ループを防止
    
    - Claudeの「思考」機能: 
      * planner_model="claude-3-7-sonnet-latest"の場合に有効化
      * モデルがトークン予算内で内部推論プロセスを実行
      * より深い分析と評価が可能
    
    Args:
        state (SectionState): セクション処理の現在状態
            - topic: レポート全体のテーマ
            - section: 処理対象のセクション情報（名前、説明、現在の内容など）
            - source_str: 検索で取得した情報ソース
            - search_iterations: 現在の検索反復回数
            
        config (RunnableConfig): ワークフロー設定情報
            - max_search_depth: 最大検索反復回数（デフォルト: 2）
            - writer_provider/model: セクション執筆に使用するAIモデル情報
            - planner_provider/model: 評価に使用するAIモデル情報
        
    Returns:
        Command: ワークフロー制御コマンド。以下のいずれかの形式:
            - 完了時: Command(update={"completed_sections": [section]}, goto=END)
            - 追加調査時: Command(update={"search_queries": new_queries, "section": updated_section}, goto="search_web")
    
    例:
    入力: 
      state={"topic": "量子コンピューティング", "section": Section(...), "source_str": "検索結果...", "search_iterations": 0}
      
    完了時の返値: 
      Command(update={"completed_sections": [完成したセクション]}, goto=END)
      
    追加調査時の返値: 
      Command(update={"search_queries": [SearchQuery(...), ...], "section": 更新されたセクション}, goto="search_web")
    """

    # 1. 現在の状態から必要な情報を取得
    topic = state["topic"]         # レポート全体のテーマ
    section = state["section"]     # 処理対象のセクションオブジェクト
    source_str = state["source_str"]  # 検索で取得した情報（ウェブコンテンツ）

    # 2. 設定情報を取得
    # LangGraphの設定からConfiguration構造体を作成
    configurable = Configuration.from_runnable_config(config)

    # 3. セクション執筆用のプロンプトを準備
    # 執筆プロンプトには以下の要素を統合:
    # - レポートのトピック（全体テーマ）
    # - セクション名と詳細説明
    # - 検索結果コンテンツ
    # - 既存のセクション内容（反復実行時に存在）
    section_writer_inputs_formatted = section_writer_inputs.format(
        topic=topic,                   # レポート全体のテーマ
        section_name=section.name,     # セクション名（例: "量子アルゴリズムの応用"）
        section_topic=section.description,  # セクションの詳細説明
        context=source_str,            # 検索結果テキスト（情報源）
        section_content=section.content   # 既存内容（あれば）- 反復実行時に使用
    )

    # 4. セクション執筆用のAIモデルを準備
    # 設定されたプロバイダとモデルでLLMを初期化
    writer_provider = get_config_value(configurable.writer_provider)  # モデルプロバイダ（OpenAI, Anthropicなど）
    writer_model_name = get_config_value(configurable.writer_model)   # モデル名（GPT-4, Claude-3.5など）
    writer_model = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider
    ) 

    # 5. AIモデルを使ってセクションを執筆
    # 2段階プロンプト構造（システム指示 + 具体的な入力）でセクション内容を生成
    section_content = writer_model.invoke([
        SystemMessage(content=section_writer_instructions),  # セクション執筆の一般指示
        HumanMessage(content=section_writer_inputs_formatted)  # このセクション固有の情報
    ])
    
    # 6. 生成された内容をセクションオブジェクトに格納
    # モデルからの応答をセクションの内容として保存
    section.content = section_content.content

    # 7. セクション評価用のプロンプト準備
    # セクションの品質評価と追加情報の必要性を判断するためのプロンプト
    section_grader_message = ("レポートを評価し、不足している情報に関するフォローアップ質問を検討してください。"
                             "評価が「合格」の場合は、すべてのフォローアップクエリに空の文字列を返してください。"
                             "評価が「不合格」の場合は、不足している情報を収集するための具体的な検索クエリを提供してください。")
    
    # セクション評価のための詳細指示を準備
    # 評価基準と対象セクションの情報を含むプロンプトを生成
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic,                # レポートテーマ
        section_topic=section.description,  # セクションテーマ
        section=section.content,     # 評価対象の生成されたセクション内容
        number_of_follow_up_queries=configurable.number_of_queries  # 生成するフォローアップクエリ数
    )

    # 8. 評価用のモデル準備（計画モデルを使用）
    # セクション執筆とは別のモデルで評価することで客観性を高める
    planner_provider = get_config_value(configurable.planner_provider)  # 評価モデルのプロバイダ
    planner_model = get_config_value(configurable.planner_model)  # 評価モデル名

    # Claude 3.7 Sonnetの場合は特別な「思考」設定を適用
    # 「思考」機能により、モデルはより深い分析と評価が可能
    if planner_model == "claude-3-7-sonnet-latest":
        # 思考プロセスのためのトークン予算を配分
        # max_tokens: 出力の最大トークン数
        # thinking: 内部思考プロセスの有効化と予算設定
        reflection_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider, 
            max_tokens=20_000,  # 出力の最大トークン数（十分な長さを確保）
            thinking={"type": "enabled", "budget_tokens": 16_000}  # 思考に割り当てるトークン数
        ).with_structured_output(Feedback)  # 構造化出力でFeedback型を強制
    else:
        # その他のモデルは標準設定で初期化
        # Claude 3.7以外は思考機能なしで評価を実行
        reflection_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider
        ).with_structured_output(Feedback)  # 構造化出力を設定

    # 9. セクションの品質評価を実行
    # 評価AIにセクション内容を評価させ、改善点や追加調査が必要な点を特定
    feedback = reflection_model.invoke([
        SystemMessage(content=section_grader_instructions_formatted),  # 評価指示
        HumanMessage(content=section_grader_message)  # 評価依頼メッセージ
    ])
    # 返されるfeedbackは以下の構造:
    # - grade: "pass"（合格）または"fail"（不合格）
    # - follow_up_queries: 追加調査が必要な検索クエリのリスト

    # 10. 評価結果に基づく分岐処理
    # 10-A: セクションが合格（評価=pass）または最大検索回数に達した場合
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # セクションを完了済みとしてマークし、処理を終了
        # update: completed_sectionsリストにこのセクションを追加
        # goto: END - このセクションの処理を終了し、次のノードに進む
        return Command(
            update={"completed_sections": [section]},  # 完了セクションリストに追加
            goto=END  # 処理終了を指示（このセクションの調査・執筆サイクルを終了）
        )
    # 10-B: 品質が不十分（評価=fail）でさらに検索が必要な場合
    else:
        # 評価で生成された新しい検索クエリを使って検索プロセスに戻る
        # update: 新しい検索クエリと最新のセクション内容を設定
        # goto: "search_web" - 検索ノードに戻って反復処理を継続
        return Command(
            update={
                "search_queries": feedback.follow_up_queries,  # 評価で提案された新しい検索クエリ
                "section": section  # 現在のセクション内容（部分的に完成している状態）
            },
            goto="search_web"  # 検索ノードに戻り、新しいクエリで追加情報を取得
        )
    
def write_final_sections(state: SectionState, config: RunnableConfig):
    """研究不要の最終セクション（はじめにや結論など）を執筆する関数
    
    この関数は、直接的なウェブ調査を必要としない「はじめに」や「結論」などの
    特殊セクションを担当します。これらのセクションは、レポート全体の品質と
    一貫性を高める重要な役割を果たします。この関数は調査セクションが完了した
    後に実行され、すべての調査内容を把握した上で執筆されます。
    
    詳細な処理フロー:
    1. 設定情報の取得:
       - LangGraph設定から執筆モデル情報を取得
       - writer_provider/writer_modelパラメータを正規化
       - 例: writer_provider="anthropic", writer_model="claude-3-5-sonnet-latest"
       
    2. 執筆に必要なデータの準備:
       - レポート全体のトピック（例: "自動運転技術の現状と未来"）
       - 執筆対象セクション情報（名前、説明など）
       - 調査済みセクションの内容（本文部分のコンテンツ）
       - 例: section.name="はじめに" または "結論"
       
    3. セクション特性に応じたプロンプト生成:
       - final_section_writer_instructionsテンプレートを使用
       - セクションタイプ（はじめに/結論）に応じた特化指示
       - セクション特有の要件（構造、長さ、スタイル）を含む
       - 例: はじめには50-100語、結論には100-150語の制限
       
    4. 執筆用AIモデルの初期化:
       - 設定からプロバイダとモデル名を取得してモデルを準備
       - LangChainのinit_chat_modelでモデルを初期化
       - 例: Claude 3.5 Sonnetで自然な文章スタイルを実現
       
    5. セクション執筆の実行:
       - システムプロンプトとシンプルな人間メッセージでモデルを呼び出し
       - はじめにの場合: レポートの背景、目的、重要性を簡潔に説明
       - 結論の場合: 主要な発見を要約し、インサイトや今後の展望を提供
       - 表やリストなどの構造要素も必要に応じて含む
       
    6. 結果の整形と返却:
       - 生成されたコンテンツをセクションオブジェクトに格納
       - 完了したセクションを「completed_sections」リストに追加して返却
    
    セクション別の特別処理:
    - 「はじめに」セクション:
       * #で始まるタイトル（Markdown H1形式）
       * 50-100語の簡潔な文章
       * レポートの目的と重要性を強調
       * 構造要素（リスト、表）は使用しない
       
    - 「結論」セクション:
       * ##で始まるタイトル（Markdown H2形式）
       * 100-150語の要約文
       * 比較レポートの場合はMarkdownテーブルを含む
       * 非比較レポートの場合は適切な箇条書きを含む場合がある
       * 全体のまとめと洞察・次のステップを提供
    
    技術的詳細:
    - 「はじめに」と「結論」はレポート全体の印象に大きく影響するため、高品質なモデルを使用
    - section.researchフラグがFalseのセクションがこの関数で処理される
    - 複数の最終セクションが存在する場合、それぞれ独立した呼び出しで処理
    
    Args:
        state (SectionState): セクション執筆に関する現在の状態
            - topic: レポートのメインテーマ（例: "ブロックチェーン技術の応用"）
            - section: 執筆対象のセクション情報（Section型オブジェクト）
                - name: セクション名（"はじめに"や"結論"など）
                - description: セクションの詳細説明
                - research: 調査フラグ（この関数ではFalse）
            - report_sections_from_research: 調査済み本文セクションの内容（統合テキスト）
                
        config (RunnableConfig): LangGraphの実行時設定情報
            - writer_provider: 文章生成モデルのプロバイダ
            - writer_model: 文章生成に使用するモデル名
        
    Returns:
        Dict[str, List[Section]]: 完了したセクションを含む辞書
            - "completed_sections": 完了したセクションオブジェクトのリスト（長さ1）
                - セクションオブジェクトには生成されたコンテンツがセットされている
    
    例:
    入力: 
      state={
        "topic": "気候変動対策", 
        "section": Section(name="はじめに", description="レポートの目的", research=False),
        "report_sections_from_research": "セクション1: 現状分析\n温室効果ガスの削減は..."
      }
      
    返値: 
      {"completed_sections": [Section(name="はじめに", description="レポートの目的", content="# 気候変動対策\n\n本レポートでは...", research=False)]}
    """

    # 1. 設定情報の取得
    # LangGraph設定からConfiguration構造体を作成し、執筆モデル情報を取得
    configurable = Configuration.from_runnable_config(config)

    # 2. 現在の状態から必要なデータを抽出
    topic = state["topic"]  # レポートのメインテーマ（例: "量子コンピューティングの進化"）
    section = state["section"]  # 執筆対象のセクション（はじめに/結論など）
    completed_report_sections = state["report_sections_from_research"]  # 調査済み本文セクションの内容
    
    # 3. セクション執筆のシステムプロンプトを準備
    # セクションタイプによって異なる執筆指示が含まれるプロンプトを生成
    # final_section_writer_instructionsには、はじめに/結論それぞれの執筆ガイドラインが含まれている
    system_instructions = final_section_writer_instructions.format(
        topic=topic,  # レポートテーマ（例: "再生可能エネルギーの未来"）
        section_name=section.name,  # セクション名（"はじめに"か"結論"など）
        section_topic=section.description,  # セクションの説明（例: "レポートの概要と目的"）
        context=completed_report_sections  # 調査済みセクションの内容（参照用）
    )

    # 4. 執筆用のAIモデルを準備
    # 設定されたプロバイダとモデルでAIを初期化
    writer_provider = get_config_value(configurable.writer_provider)  # モデルプロバイダ（OpenAI, Anthropicなど）
    writer_model_name = get_config_value(configurable.writer_model)  # モデル名（GPT-4, Claude-3.5など）
    writer_model = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider
    ) 
    
    # 5. AIモデルを使ってセクションを執筆
    # システムプロンプトとシンプルな人間メッセージでモデルを呼び出し
    # モデルは調査セクションの内容を理解し、適切な導入/結論を生成
    section_content = writer_model.invoke([
        SystemMessage(content=system_instructions),  # セクション別の詳細な指示
        HumanMessage(content="Generate a report section based on the provided sources.")  # シンプルな指示
    ])
    
    # 6. 生成されたテキストをセクションオブジェクトに格納
    # モデルの応答をセクションの内容フィールドにセット
    section.content = section_content.content

    # 7. 執筆完了したセクションを返却
    # 完了したセクションを含む辞書を返却（gather_completed_sectionsで後で処理される）
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """完了した調査セクションを収集し、最終セクション執筆用のコンテキストとして整形する関数
    
    この関数は、レポート作成プロセスの重要な「ブリッジ」として機能します。並行して
    処理された各調査セクションの結果を一つに統合し、「はじめに」や「結論」などの
    最終セクション執筆のためのコンテキストを準備します。これにより、レポート全体の
    一貫性と論理的つながりが確保されます。
    
    詳細な処理フロー:
    1. 完了セクションの収集:
       - 状態から完了したセクション（research=Trueで調査済み）のリストを取得
       - これらは並行して調査・執筆された複数のセクションオブジェクト
       - 例: セクション「市場分析」「技術動向」「競合状況」など
       
    2. セクションの整形と統合:
       - format_sections関数を使って全セクションを構造化テキストに変換
       - 各セクションは区切り線で明確に分離され、以下の情報を含む:
         * セクション名（例: "Section 1: 市場分析"）
         * 説明（例: "Description: AIツール市場の現状と予測"）
         * 調査フラグ（例: "Requires Research: True"）
         * 内容（例: "## 市場分析\nAIツール市場は2023年に..."）
       - 整形例:
         ```
         ==========
         Section 1: 市場分析
         ==========
         Description: AIツール市場の現状と予測
         Requires Research: True
         Content:
         ## 市場分析
         AIツール市場は2023年に約500億ドルの規模に達し...
         ```
       
    3. コンテキスト情報の生成:
       - 整形されたテキストを「report_sections_from_research」キーとして状態に追加
       - この情報は後続の「write_final_sections」ノードで以下の用途に使用:
         * 「はじめに」の執筆: レポート全体の概要と目的の紹介
         * 「結論」の執筆: 調査結果の統合と全体的な洞察の提供
    
    技術的詳細:
    - この関数は「同期ポイント」として機能: 並行処理されたセクションが揃ったことを確認
    - format_sections関数: セクションを視覚的に区別しやすい構造化テキストに変換
    - LangGraphの状態管理: 新しい状態キーを追加することで次のノードに情報を引き渡し
    
    Args:
        state (ReportState): レポート作成の現在状態
            - completed_sections: 完了したSection型オブジェクトのリスト
              各オブジェクトには名前、説明、調査フラグ、生成された内容が含まれる
        
    Returns:
        Dict[str, str]: 整形された調査セクション内容を含む辞書
            - "report_sections_from_research": すべての完了セクションを整形した文字列
              この値は後続の最終セクション執筆ノードでコンテキストとして使用される
    
    例:
    入力: 
      state={"completed_sections": [
        Section(name="市場分析", description="AI市場の現状", content="## 市場分析\nAI市場は...", research=True),
        Section(name="技術動向", description="最新AI技術", content="## 技術動向\n大規模言語モデルは...", research=True)
      ]}
      
    返値: 
      {"report_sections_from_research": "==========\nSection 1: 市場分析\n==========\nDescription: AI市場の現状\n..."}
    """

    # 1. 状態から完了したセクションのリストを取得
    # これらは並行して処理された各調査セクション（research=True）のオブジェクト
    # 各セクションには生成されたコンテンツが格納されている
    completed_sections = state["completed_sections"]

    # 2. 完了したセクション群を一つの統合テキストに整形
    # format_sections関数はセクションリストを受け取り、読みやすく構造化されたテキストを返す
    # 各セクションは明確な区切りと一貫した形式で表示される
    completed_report_sections = format_sections(completed_sections)
    # 変換例:
    # ======================================================
    # Section 1: 市場規模と成長
    # ======================================================
    # Description: 2025年および2030年の市場規模、成長率(CAGR)や主要地域...
    # Requires Research: True
    #
    # Content:
    # ## 市場規模と成長
    # 2025年のAI市場規模は約3,060億ドル、2030年には約1兆3,391億ドル...
    # ...

    # 3. 整形したテキストを状態に追加して返却
    # この値は後続の「write_final_sections」ノードでコンテキストとして使用され、
    # 「はじめに」や「結論」などの最終セクション執筆の基礎となる
    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState):
    """すべてのセクションを最終的なレポートとして統合する関数
    
    この関数はレポート生成プロセスの最終ステップとして、すべての完成セクションを
    元のレポート計画で指定された順序で配置し、一つの完全なドキュメントとして統合します。
    これにより、一貫性のある論理的な流れを持つ最終レポートが生成されます。
    
    詳細な処理フロー:
    1. レポート構造の取得:
       - レポート計画段階で定義されたオリジナルのセクション順序を取得
       - この順序情報が最終レポートの論理的な流れを決定する
       - 例: [はじめに] → [背景] → [分析] → [ケーススタディ] → [結論]
       
    2. 完了セクションのマッピング:
       - 完了したすべてのセクション（調査セクション + 最終セクション）を取得
       - セクション名をキーとする辞書に変換して高速アクセスを可能に
       - 例: {"はじめに": "# レポートタイトル\n...", "背景": "## 背景\n..."}
       
    3. セクション内容の組み立て:
       - 元のセクション順序を保持しながら、各セクションに完成した内容を統合
       - このステップにより、計画時に設計した論理的な流れが維持される
       - 例: Section(name="はじめに").content = "# レポートタイトル\n..."
       
    4. 最終ドキュメント生成:
       - 順序付けられたセクションの内容を連結して最終レポートを作成
       - 適切な間隔（改行）を挿入して読みやすさを確保
       - Markdown形式のフォーマットを維持（見出し、リスト、表など）
    
    技術的詳細:
    - セクション順序の保存: レポート計画段階の順序が最終出力に反映される
    - 辞書によるO(1)アクセス: 名前をキーとする辞書でセクション内容への高速アクセスを実現
    - state["sections"]とstate["completed_sections"]の関係:
      * sections: 元のレポート計画で定義されたセクション（順序情報あり）
      * completed_sections: 執筆が完了した各セクション（順序情報なし）
    
    Args:
        state (ReportState): 現在のレポート状態
            - sections: オリジナルのレポート計画のセクションリスト（順序情報あり）
            - completed_sections: 執筆が完了したすべてのセクションのリスト
              （調査セクションと最終セクションの両方を含む）
        
    Returns:
        Dict[str, str]: 最終レポートを含む辞書
            - "final_report": 完全なレポートテキスト（Markdown形式）
    
    例:
    入力: 
      state={
        "sections": [Section(name="はじめに"), Section(name="分析"), Section(name="結論")],
        "completed_sections": [
          Section(name="分析", content="## 分析\n詳細な分析結果..."),
          Section(name="はじめに", content="# レポートタイトル\n本レポートでは..."),
          Section(name="結論", content="## 結論\n分析の結果...")
        ]
      }
      
    返値: 
      {"final_report": "# レポートタイトル\n本レポートでは...\n\n## 分析\n詳細な分析結果...\n\n## 結論\n分析の結果..."}
    """

    # 1. 状態から必要な情報を取得
    sections = state["sections"]  # オリジナルのレポート計画のセクション（順序情報あり）
    completed_sections = {s.name: s.content for s in state["completed_sections"]}  # 完了したセクション（名前:内容のマップ）

    # 2. オリジナルの順序を維持しながら各セクションに完成した内容を入れる
    # レポート計画で定義した順序を保持したまま、各セクションに執筆済みの内容を挿入
    # これにより、ランダムに完了した各セクションが適切な順序で配置される
    for section in sections:
        # 完了セクションの辞書から、このセクション名に対応する内容を取得
        section.content = completed_sections[section.name]

def initiate_final_section_writing(state: ReportState):
    """調査不要セクション（はじめにや結論）の執筆タスクを並行して開始する関数
    
    この関数はLangGraphの「条件付きエッジ関数」として動作し、レポート生成ワークフローの
    重要な分岐点を担当します。本文調査が完了した後、次のステップとして調査を必要としない
    セクション（はじめに、結論など）の執筆タスクを設定し、並行処理のためのSend命令を生成します。
    
    詳細な処理フロー:
    1. 調査不要セクションの識別:
       - オリジナルレポート計画（state["sections"]）から調査不要セクション（research=False）を抽出
       - 典型的なresearch=Falseセクションには以下が含まれる:
         * 「はじめに」/「イントロダクション」: レポートの目的と背景を説明
         * 「結論」/「サマリー」: 調査結果の総括と洞察を提供
         * 「エグゼクティブサマリー」: 経営層向けの要約（計画に含まれる場合）
       - 例: レポート計画に「はじめに」と「結論」の2つのresearch=Falseセクションがある場合、
         2つの並行タスクが生成される
       
    2. タスク命令の生成:
       - 各調査不要セクションごとにLangGraphのSend命令を作成
       - Send命令は以下の要素で構成:
         * 送信先ノード: "write_final_sections"（最終セクション執筆専用ノード）
         * 送信データ: タスク実行に必要なコンテキスト情報の辞書
       - 各タスクには以下の情報が含まれる:
         * topic: レポート全体のテーマ（例: "AIの倫理的課題"）
         * section: 執筆対象のセクションオブジェクト（詳細情報を含む）
         * report_sections_from_research: 調査済みセクションの統合内容（執筆の参照資料）
       
    3. 並行処理の指示:
       - 生成されたSend命令のリストを返却
       - LangGraphエンジンがこれを解釈し、複数のセクション執筆タスクを並行して起動
       - これにより複数の最終セクションを効率的に処理（例: はじめにと結論を同時執筆）
    
    技術的詳細:
    - 条件付きエッジ関数: LangGraphグラフで「next」エッジからの遷移を動的に制御
    - Send命令: 並行処理タスクを指定するLangGraph特有の命令形式
      * "write_final_sections"ノードを呼び出すが、別々のセクションデータで複数回実行
    - リスト内包表記: PythonのリストコンプリヘンションでSend命令リストを効率的に生成
    - 条件フィルタリング: `if not s.research`でresearch=Falseのセクションのみを処理
    
    Args:
        state (ReportState): レポートの現在状態
            - sections: レポート計画の全セクションリスト（research=TrueとFalseの両方を含む）
            - topic: レポートのメインテーマ
            - report_sections_from_research: 調査済みセクションの統合コンテンツ
                （gather_completed_sections関数で生成）
        
    Returns:
        List[Send]: 最終セクション執筆の並行タスクを指示するSend命令のリスト
            - 各Send命令は"write_final_sections"ノードを単一セクションのデータで呼び出す
            - 返却リストの長さはresearch=Falseセクションの数と同じ（通常は2-3）
    
    例:
    入力状態: 
      state={
        "sections": [
          Section(name="はじめに", research=False), 
          Section(name="市場分析", research=True),
          Section(name="結論", research=False)
        ],
        "topic": "AI市場の動向",
        "report_sections_from_research": "== 市場分析 ==\n2023年のAI市場は..."
      }
      
    返値: 
      [
        Send("write_final_sections", {
          "topic": "AI市場の動向", 
          "section": Section(name="はじめに", research=False),
          "report_sections_from_research": "== 市場分析 ==\n2023年のAI市場は..."
        }),
        Send("write_final_sections", {
          "topic": "AI市場の動向",
          "section": Section(name="結論", research=False),
          "report_sections_from_research": "== 市場分析 ==\n2023年のAI市場は..."
        })
      ]
    
    ワークフロー上の位置づけ:
    - 前ノード: gather_completed_sections（調査セクションの統合）
    - 次ノード: write_final_sections（最終セクションの執筆）
    - この関数により、本文調査→最終セクション執筆への橋渡しが行われる
    """

    # 1. 調査不要セクションの抽出と並行タスクの生成
    # このリスト内包表記は以下の処理を行います:
    # - レポート計画の全セクション（state["sections"]）をループ
    # - 各セクション(s)のresearchフラグをチェック
    # - research=Falseのセクションのみに対してSend命令を生成
    # - 結果として、調査不要セクションごとに1つのSend命令を含むリストを返却
    return [
        # Send命令の構造:
        # - 第1引数: 送信先ノード名（"write_final_sections"）
        # - 第2引数: 送信データ（辞書形式）- 必要なコンテキスト情報を含む
        Send("write_final_sections", {
            "topic": state["topic"],  # レポート全体のテーマ（例: "量子コンピューティングの進展"）
            "section": s,  # 執筆対象の特定のセクションオブジェクト（例: はじめに/結論）
            "report_sections_from_research": state["report_sections_from_research"]  # 調査内容の統合テキスト
        }) 
        # リスト内包表記でsections全体をループ
        for s in state["sections"] 
        # フィルタ条件: research=Falseのセクションのみ処理
        # research=TrueのセクションはすでにParallel Research段階で処理済み
        if not s.research
    ]

# Report section sub-graph -- 
# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Outer graph for initial report plan compiling results from each section -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile()