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
    
    この関数はレポート作成プロセスの最初の重要なステップで、以下の処理を行います：
    1. 設定情報の取得と準備
    2. 計画立案に役立つ検索クエリを生成
    3. それらのクエリを使ってウェブ検索を実行
    4. 検索結果とユーザーフィードバックを基にレポート構造を計画
    
    Args:
        state (ReportState): レポートのトピックとフィードバックを含む現在の状態
        config (RunnableConfig): 使用するモデル、検索API等の設定情報
        
    Returns:
        Dict: 生成されたセクションを含む辞書
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
    
    この関数は「人間とAIの協調」を実現する重要なノードで、以下の処理を行います：
    1. 生成されたレポート計画をユーザーが確認できる形式に整形
    2. ワークフローを一時停止（interrupt）してユーザーからのフィードバックを取得
    3. フィードバックに基づいて次のステップを決定：
       - 計画が承認された場合：各セクションの調査・執筆を並行して開始
       - フィードバックが提供された場合：そのフィードバックを基に計画を再生成
    
    Args:
        state (ReportState): レビュー対象のセクションを含む現在のグラフ状態
        config (RunnableConfig): ワークフロー設定情報（この関数では直接使用しない）
        
    Returns:
        Command: 次に実行するノードと更新状態を指定するコマンドオブジェクト
    """

    # 1. 現在のレポート計画をユーザー表示用に整形
    topic = state["topic"]  # レポートのメインテーマ
    sections = state['sections']  # 生成されたセクションリスト
    
    # セクションを読みやすい形式に変換
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"  # セクション名
        f"Description: {section.description}\n"  # セクションの説明
        f"Research needed: {'Yes' if section.research else 'No'}\n"  # 調査が必要かどうか
        for section in sections
    )

    # 2. ユーザーにレポート計画を提示し、フィードバックを要求
    # interrupt関数はワークフローを一時停止し、UIでユーザーにメッセージを表示
    interrupt_message = f"""以下のレポート計画についてフィードバックをお願いします。
                        \n\n{sections_str}\n
                        \nこのレポート計画はあなたのニーズを満たしていますか？\n計画を承認する場合は「true」を入力してください。\nまたは、レポート計画を再生成するためのフィードバックを提供してください："""
    
    # フィードバックを待機（ユーザーが応答するまでここで処理が止まる）
    feedback = interrupt(interrupt_message)

    # 3. ユーザーのフィードバックに基づいて処理を分岐
    # 3-A: ユーザーが計画を承認した場合（True が返された）
    if isinstance(feedback, bool) and feedback is True:
        # 調査が必要なセクションそれぞれについて、並行して処理を開始
        return Command(goto=[
            # 各セクションに対してSend命令を作成（並列処理の指示）
            Send("build_section_with_web_research", {
                "topic": topic,  # レポートのトピック
                "section": s,    # 処理対象のセクション
                "search_iterations": 0  # 検索反復回数の初期値
            }) 
            # sections内の各要素sに対して反復するが、research=Trueのセクションのみ処理
            for s in sections 
            if s.research
        ])
    
    # 3-B: ユーザーが修正フィードバックを提供した場合（文字列が返された）
    elif isinstance(feedback, str):
        # フィードバックを状態に保存して計画生成ノードに戻る
        return Command(
            goto="generate_report_plan",  # 計画生成ノードに移動
            update={"feedback_on_report_plan": feedback}  # 状態にフィードバックを追加
        )
    
    # 3-C: 想定外の型のフィードバックが返された場合（エラー処理）
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
def generate_queries(state: SectionState, config: RunnableConfig):
    """特定のセクションの調査用検索クエリを生成するノード
    
    セクションのトピックと説明に基づいて、対象を絞った検索クエリを生成します。
    
    Args:
        state: セクション詳細を含む現在の状態
        config: 生成するクエリの数などの設定
        
    Returns:
        生成された検索クエリを含む辞書
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(topic=topic, 
                                                           section_topic=section.description, 
                                                           number_of_queries=number_of_queries)

    # Generate queries  
    queries = structured_llm.invoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """セクションのクエリに対してウェブ検索を実行するノード
    
    このノード：
    1. 生成されたクエリを取得
    2. 設定された検索APIを使用して検索を実行
    3. 結果を使いやすい形式に整形
    
    Args:
        state: 検索クエリを含む現在の状態
        config: 検索API設定
        
    Returns:
        検索結果と更新された反復カウントを含む辞書
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """レポートのセクションを執筆し、品質評価によって追加調査の必要性を判断する関数
    
    この関数はレポート生成プロセスの中核で、以下の重要なステップを実行します：
    1. 検索結果を使ってセクションの内容を執筆
    2. 執筆された内容の品質を評価
    3. 評価結果に基づいて次のアクションを決定:
       - 高品質であれば：セクションを完了とし次のステップへ
       - 不十分であれば：追加情報のための新しい検索を実行
    
    この「執筆→評価→改善」のサイクルにより、高品質な内容を保証します。
    
    Args:
        state (SectionState): 検索結果とセクション情報を含む現在の状態
        config (RunnableConfig): 執筆モデルと評価の設定情報
        
    Returns:
        Command: セクション完了または追加検索のコマンド
    """

    # 1. 現在の状態から必要な情報を取得
    topic = state["topic"]
    section = state["section"]
    # 検索で取得した情報ソース
    source_str = state["source_str"]

    # 2. 設定情報を取得
    configurable = Configuration.from_runnable_config(config)

    # 3. セクション執筆用のプロンプトを準備
    # 以下の情報を組み合わせて適切なプロンプトを生成:
    # - レポートトピック
    # - セクション名
    # - セクションのトピック説明
    # - 検索で得た情報
    # - 既存のセクション内容（反復検索の場合）
    section_writer_inputs_formatted = section_writer_inputs.format(
        topic=topic, 
        section_name=section.name, 
        section_topic=section.description, 
        context=source_str, 
        section_content=section.content
    )

    # 4. セクション執筆用のAIモデルを準備
    writer_provider = get_config_value(configurable.writer_provider)  # OpenAI、Anthropicなど
    writer_model_name = get_config_value(configurable.writer_model)   # GPT-4、Claude-3.5など
    writer_model = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider
    ) 

    # 5. AIモデルを使ってセクションを執筆
    section_content = writer_model.invoke([
        SystemMessage(content=section_writer_instructions),  # 一般的な執筆指示
        HumanMessage(content=section_writer_inputs_formatted)  # 具体的な執筆内容
    ])
    
    # 6. 生成された内容をセクションオブジェクトに格納
    section.content = section_content.content

    # 7. セクション評価用のプロンプト準備
    # セクションの品質評価と不足情報の特定のためのメッセージ
    section_grader_message = ("レポートを評価し、不足している情報に関するフォローアップ質問を検討してください。"
                             "評価が「合格」の場合は、すべてのフォローアップクエリに空の文字列を返してください。"
                             "評価が「不合格」の場合は、不足している情報を収集するための具体的な検索クエリを提供してください。")
    
    # セクション評価のための詳細な指示を準備
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic, 
        section_topic=section.description,
        section=section.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )

    # 8. 評価用のモデル準備（計画モデルを使用）
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # Claude 3.7 Sonnetの場合は特別な「思考」設定を適用
    if planner_model == "claude-3-7-sonnet-latest":
        # 思考プロセスのためのトークン予算を配分
        reflection_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider, 
            max_tokens=20_000,  # 出力の最大トークン数
            thinking={"type": "enabled", "budget_tokens": 16_000}  # 思考に割り当てるトークン
        ).with_structured_output(Feedback)
    else:
        # その他のモデルは標準設定
        reflection_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider
        ).with_structured_output(Feedback)

    # 9. セクションの品質評価を実行
    # Feedbackクラスの構造化出力を得る（grade, follow_up_queriesフィールドを含む）
    feedback = reflection_model.invoke([
        SystemMessage(content=section_grader_instructions_formatted),
        HumanMessage(content=section_grader_message)
    ])

    # 10. 評価結果に基づく分岐処理
    # 10-A: セクションが合格（評価=pass）または最大検索回数に達した場合
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # セクションを完了済みとしてマークし、処理を終了
        return Command(
            update={"completed_sections": [section]},  # 完了セクションに追加
            goto=END  # 処理終了を指示
        )
    # 10-B: 品質が不十分（評価=fail）でさらに検索が必要な場合
    else:
        # 評価で生成された新しい検索クエリを使って検索プロセスに戻る
        return Command(
            update={
                "search_queries": feedback.follow_up_queries,  # 新しい検索クエリをセット
                "section": section  # 最新のセクション内容を保持
            },
            goto="search_web"  # 検索ノードに戻る
        )
    
def write_final_sections(state: SectionState, config: RunnableConfig):
    """研究不要の最終セクション（はじめにや結論など）を執筆する関数
    
    この関数は、直接的な調査を必要としないセクションを担当します。具体的には：
    1. 「はじめに」- レポート全体の概要と目的を紹介
    2. 「結論」- 調査済みセクションの内容を統合して総括
    
    これらのセクションは、すでに調査・執筆された本文セクションの内容に基づいて
    作成されるため、新たな調査は不要で、既存コンテンツの合成と要約が中心となります。
    
    Args:
        state (SectionState): 以下を含む現在の状態
            - topic: レポートのメインテーマ
            - section: 執筆対象のセクション
            - report_sections_from_research: 調査済みセクションの内容（コンテキスト）
        config (RunnableConfig): 執筆モデルの設定情報
        
    Returns:
        Dict: 新しく執筆されたセクションを含む辞書（完了セクションリストに追加される）
    """

    # 1. 設定情報の取得
    configurable = Configuration.from_runnable_config(config)

    # 2. 現在の状態から必要なデータを抽出
    topic = state["topic"]  # レポートのメインテーマ
    section = state["section"]  # 執筆対象のセクション（はじめに/結論など）
    completed_report_sections = state["report_sections_from_research"]  # 調査済み本文セクションの内容
    
    # 3. セクション執筆のシステムプロンプトを準備
    # final_section_writer_instructionsには、はじめに/結論それぞれに特化した指示が含まれている
    system_instructions = final_section_writer_instructions.format(
        topic=topic,  # レポートテーマ
        section_name=section.name,  # セクション名（「はじめに」か「結論」など）
        section_topic=section.description,  # セクションの説明
        context=completed_report_sections  # 調査済みセクションの内容（参照用）
    )

    # 4. 執筆用のAIモデルを準備
    writer_provider = get_config_value(configurable.writer_provider)  # OpenAI、Anthropicなど
    writer_model_name = get_config_value(configurable.writer_model)  # GPT-4、Claude-3.5など
    writer_model = init_chat_model(
        model=writer_model_name, 
        model_provider=writer_provider
    ) 
    
    # 5. AIモデルを使ってセクションを執筆
    section_content = writer_model.invoke([
        SystemMessage(content=system_instructions),  # セクション特有の指示
        HumanMessage(content="Generate a report section based on the provided sources.")  # 簡潔な指示
    ])
    
    # 6. 生成されたテキストをセクションオブジェクトに格納
    section.content = section_content.content

    # 7. 執筆完了したセクションを返却（完了セクションリストに追加）
    return {"completed_sections": [section]}

def gather_completed_sections(state: ReportState):
    """完了した調査セクションを収集し、最終セクション執筆用のコンテキストとして整形する関数
    
    この関数は、レポート作成フローにおける「橋渡し」の役割を果たします。具体的には：
    1. これまでに調査・執筆が完了したすべてのセクションを収集
    2. それらを一つの統合されたテキストに整形
    3. このテキストを「はじめに」や「結論」など調査不要セクションの執筆時のコンテキストとして提供
    
    これにより、調査セクションで得られた知見が最終セクションに反映され、
    レポート全体として一貫性のある文書になります。
    
    Args:
        state (ReportState): 完了したセクションを含む現在のグラフ状態
        
    Returns:
        Dict: 整形されたセクション内容を含む辞書
    """

    # 1. 状態から完了したセクションのリストを取得
    # これらは並行して調査・執筆された各セクションオブジェクト
    completed_sections = state["completed_sections"]

    # 2. 完了したセクション群を一つの文字列に整形
    # format_sections関数は、セクションの内容を読みやすい構造化テキストに変換
    completed_report_sections = format_sections(completed_sections)
    # 変換例：
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
    # この値は後続の「write_final_sections」ノードでコンテキストとして使用される
    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState):
    """すべてのセクションを最終的なレポートとして統合する関数
    
    この関数はレポート生成プロセスの最終ステップとして、以下の役割を果たします：
    1. すべての完成したセクション（調査セクションと最終セクション）を収集
    2. 元のレポート計画で指定された順序を維持
    3. セクションを連結して一つの完全なレポートを生成
    
    これにより、段階的に作成された様々なセクションが一つの統合された文書に完成します。
    
    Args:
        state (ReportState): すべての完了セクションとオリジナルの計画を含む状態
        
    Returns:
        Dict: 完成したレポート全文を含む辞書
    """

    # 1. 状態から必要な情報を取得
    sections = state["sections"]  # オリジナルのレポート計画のセクション（順序情報あり）
    completed_sections = {s.name: s.content for s in state["completed_sections"]}  # 完了したセクション（名前:内容のマップ）

    # 2. オリジナルの順序を維持しながら各セクションに完成した内容を入れる
    # これにより、計画段階で決めたセクションの順序を保持したまま内容を更新
    for section in sections:
        section.content = completed_sections[section.name]

    # 3. すべてのセクションを連結して最終レポートを作成
    # 各セクションの内容を改行で区切って連結
    all_sections = "\n\n".join([s.content for s in sections])

    # 4. 完成したレポート全文を返却
    return {"final_report": all_sections}

def initiate_final_section_writing(state: ReportState):
    """調査不要セクション（はじめにや結論）の執筆タスクを並行して開始する関数
    
    この関数はLangGraphの「エッジ関数」として動作し、以下の役割を果たします：
    1. オリジナルレポート計画から調査を必要としないセクション（research=False）を特定
    2. これらのセクション（主に「はじめに」と「結論」）の執筆を並行して開始
    3. 各セクションに必要なコンテキスト（調査済みセクションの内容）を提供
    
    この関数により、本文調査が終わった後に、その結果を反映した導入部と結論を
    効率的に執筆することが可能になります。
    
    Args:
        state (ReportState): セクション計画と調査結果を含む現在の状態
        
    Returns:
        List[Send]: 実行すべき並行タスクのリスト
    """

    # 調査を必要としないセクション（research=False）ごとに、執筆タスクを作成
    return [
        # 各セクションに対してwrite_final_sectionsノードへのSend命令を作成
        Send("write_final_sections", {
            "topic": state["topic"],  # レポートのメインテーマ
            "section": s,  # 執筆対象のセクション（はじめにや結論）
            "report_sections_from_research": state["report_sections_from_research"]  # 調査結果をコンテキストとして提供
        }) 
        # sections内の各要素sについて反復処理
        for s in state["sections"] 
        # ただしresearch=Falseのセクションのみを対象とする
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
