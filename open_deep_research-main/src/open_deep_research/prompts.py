report_planner_query_writer_instructions="""あなたはレポートのための調査を行っています。

<Report topic>
{topic}
</Report topic>

<Report organization>
{report_organization}
</Report organization>

<Task>
あなたの目標は、レポートセクションの計画に役立つ{number_of_queries}個のウェブ検索クエリを生成することです。

クエリは以下の条件を満たす必要があります：

1. レポートのトピックに関連していること
2. レポート構成で指定された要件を満たすのに役立つこと

レポート構造に必要な幅広い内容をカバーしながら、高品質で関連性の高いソースを見つけるのに十分な具体性を持つクエリを作成してください。
</Task>

<Format>
Queries ツールを呼び出してください
</Format>
"""

report_planner_instructions="""
簡潔で焦点の絞られたレポート計画が欲しいです。

<Report topic>
レポートのテーマは：
{topic}
</Report topic>

<Report organization>
レポートは次の構成に従う必要があります：
{report_organization}
</Report organization>

<Context>
レポートのセクションを計画するために使用するコンテキストは次のとおりです：
{context}
</Context>

<Task>
レポートのセクションのリストを作成してください。計画は重複するセクションや不要な埋め合わせがなく、簡潔で焦点を絞ったものにしてください。

例えば、良いレポート構成は次のようになります：
1/ 序論
2/ トピックAの概要
3/ トピックBの概要
4/ AとBの比較
5/ 結論

各セクションには以下のフィールドが必要です：

- Name - レポートのこのセクションの名前。
- Description - このセクションで扱う主なトピックの簡単な概要。
- Research - このセクションのウェブ調査を行うかどうか。
- Content - セクションの内容で、今は空白のままにしておきます。

統合ガイドライン：
- 例や実装の詳細は、別々のセクションではなく、主要トピックのセクション内に含めてください
- 各セクションはコンテンツの重複なく明確な目的を持つようにしてください
- 関連する概念は分離せずに組み合わせてください

提出する前に、構造に冗長なセクションがなく、論理的な流れになっているか確認してください。
</Task>

<Feedback>
レビューからのレポート構造に関するフィードバック（もしあれば）：
{feedback}
</Feedback>

<Format>
Sections ツールを呼び出してください
</Format>
"""

query_writer_instructions="""
あなたは技術レポートのセクションを作成するための包括的な情報を収集する、ターゲットを絞ったウェブ検索クエリを作成する専門技術ライターです。

<Report topic>
{topic}
</Report topic>

<Section topic>
{section_topic}
</Section topic>

<Task>
あなたの目標は、セクショントピックに関する包括的な情報を収集するのに役立つ{number_of_queries}個の検索クエリを生成することです。

クエリは以下の条件を満たす必要があります：

1. トピックに関連していること
2. トピックのさまざまな側面を検討すること

高品質で関連性の高いソースを見つけるのに十分な具体性を持つクエリを作成してください。
</Task>

<Format>
Queries ツールを呼び出してください
</Format>
"""

section_writer_instructions = """
研究レポートの一つのセクションを書いてください。

<Task>
1. レポートのトピック、セクション名、セクショントピックを注意深く確認してください。
2. 存在する場合は、既存のセクションコンテンツを確認してください。
3. 次に、提供されたソース資料を確認してください。
4. レポートセクションを書くために使用するソースを決定してください。
5. レポートセクションを書いて、ソースをリストアップしてください。
</Task>

<Writing Guidelines>
- 既存のセクションコンテンツが入力されていない場合は、一から書いてください
- 既存のセクションコンテンツが入力されている場合は、ソース資料と統合してください
- 厳密に150-200単語の制限
- シンプルで明確な言葉を使用してください
- 短い段落（最大2-3文）を使用してください
- セクションタイトルには##を使用してください（Markdown形式）
</Writing Guidelines>

<Citation Rules>
- 各固有URLにテキスト内で単一の引用番号を割り当ててください
- ### Sourcesで終わり、対応する番号で各ソースをリストアップしてください
- 重要：選択したソースに関係なく、最終リストでソースを順番に番号付けしてください（1,2,3,4...）
- 形式例：
    [1] ソースタイトル：URL
    [2] ソースタイトル：URL
</Citation Rules>

<Final Check>
1. すべての主張が提供されたソース資料に基づいていることを確認してください
2. 各URLがソースリストに一度だけ表示されることを確認してください
3. ソースが順番に番号付けされていること（1,2,3...）、ギャップがないことを確認してください
</Final Check>
"""

section_writer_inputs=""" 
<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic>
{section_topic}
</Section topic>

<Existing section content (if populated)>
{section_content}
</Existing section content>

<Source material>
{context}
</Source material>
"""

section_grader_instructions = """
レポートセクションを指定されたトピックに関して確認してください：

<Report topic>
{topic}
</Report topic>

<section topic>
{section_topic}
</section topic>

<section content>
{section}
</section content>

<task>
セクションの内容が、セクショントピックを十分に取り扱っているかを評価してください。

セクションの内容がセクショントピックを十分に取り扱っていない場合、不足している情報を収集するために{number_of_follow_up_queries}個のフォローアップ検索クエリを生成してください。
</task>

<format>
Feedbackツールを呼び出し、以下のスキーマで出力してください：

grade: Literal["pass","fail"] = Field(
    description="レスポンスが要件を満たしている（'pass'）か改訂が必要（'fail'）かを示す評価結果。"
)
follow_up_queries: List[SearchQuery] = Field(
    description="フォローアップ検索クエリのリスト。"
)
</format>
"""

final_section_writer_instructions="""
あなたはレポートの残りの部分から情報を統合するセクションを作成する専門技術ライターです。

<Report topic>
{topic}
</Report topic>

<Section name>
{section_name}
</Section name>

<Section topic> 
{section_topic}
</Section topic>

<Available report content>
{context}
</Available report content>

<Task>
1. セクション別のアプローチ：

序論の場合：
- レポートタイトルには#を使用してください（Markdown形式）
- 50-100語の制限
- シンプルで明確な言葉で書いてください
- 1-2段落でレポートの核心的な動機に焦点を当ててください
- レポートを紹介するために明確な物語の流れを使用してください
- 構造的要素は含めないでください（リストや表なし）
- ソースセクションは不要です

結論/要約の場合：
- セクションタイトルには##を使用してください（Markdown形式）
- 100-150語の制限
- 比較レポートの場合：
    * Markdownテーブル構文を使用した焦点を絞った比較表を含める必要があります
    * 表はレポートからの洞察を抽出する必要があります
    * 表のエントリは明確で簡潔にしてください
- 非比較レポートの場合：
    * レポートで述べられたポイントを抽出するのに役立つ場合にのみ、ONE構造要素を使用してください：
    * レポートに存在する項目を比較する焦点を絞った表（Markdownテーブル構文を使用）
    * または、適切なMarkdownリスト構文を使用した短いリスト：
      - 順序なしリストには`*`または`-`を使用してください
      - 順序付きリストには`1.`を使用してください
      - 適切なインデントと間隔を確保してください
- 具体的な次のステップまたは影響で終わらせてください
- ソースセクションは不要です

3. 執筆アプローチ：
- 一般的な文よりも具体的な詳細を使用してください
- 一語一語を大切にしてください
- 最も重要なポイントに集中してください
</Task>

<Quality Checks>
- 序論：50-100語の制限、レポートタイトルには#、構造的要素なし、ソースセクションなし
- 結論：100-150語の制限、セクションタイトルには##、最大でも構造要素は1つのみ、ソースセクションなし
- Markdown形式
- 回答に単語数やいかなる前文も含めないでください
</Quality Checks>
"""