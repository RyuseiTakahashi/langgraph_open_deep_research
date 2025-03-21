from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

class Section(BaseModel):
    """
    Section クラス - レポートのセクションを表現するデータモデル

    このクラスは研究レポートの一つのセクションを表し、Pydantic BaseModelを継承しています。
    レポートの構造化されたセクションを作成、管理するために使用されます。

    属性:
        name (str): このレポートセクションの名前。
        description (str): このセクションで扱われる主要なトピックと概念の簡潔な概要。
        research (bool): このレポートセクションにウェブ調査が必要かどうかを示すフラグ。
        content (str): セクションの実際のコンテンツテキスト。
        
    使用例:
        ```
        section = Section(
            name="序論",
            description="研究の背景と目的について説明します",
            research=True,
            content="この研究は..."
        ```
    """
    name: str = Field(
        description="このレポートセクションの名前。",
    )
    description: str = Field(
        description="このセクションで扱われる主要なトピックと概念の簡潔な概要。",
    )
    research: bool = Field(
        description="このセクションのレポートにウェブ調査が必要かどうか。"
    )
    content: str = Field(
        description="セクションの内容テキスト。"
    )   

class Sections(BaseModel):
    """
    Sections クラス - レポートの複数セクションを管理するデータモデル
    
    複数のセクションを一つのコンテナとして扱うためのクラスです。
    レポート全体の構造を表現します。
    """
    sections: List[Section] = Field(
        description="レポートのセクション一覧。",
    )

class SearchQuery(BaseModel):
    """
    SearchQuery クラス - ウェブ検索クエリを表現するデータモデル
    
    研究に必要な情報を取得するための検索クエリを定義します。
    """
    search_query: str = Field(None, description="ウェブ検索用のクエリ文字列。")

class Queries(BaseModel):
    """
    Queries クラス - 複数の検索クエリを管理するデータモデル
    
    複数の検索クエリをまとめて管理するためのコンテナです。
    一連の調査プロセスで使用されます。
    """
    queries: List[SearchQuery] = Field(
        description="検索クエリのリスト。",
    )

class Feedback(BaseModel):
    """
    Feedback クラス - レポートや検索結果に対するフィードバックを表現するデータモデル
    
    評価結果とフォローアップのための情報を提供します。
    品質管理と反復的な改善プロセスに使用されます。
    """
    grade: Literal["pass","fail"] = Field(
        description="応答が要件を満たしているか（'pass'）または修正が必要か（'fail'）を示す評価結果。"
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="フォローアップの検索クエリのリスト。",
    )

class ReportStateInput(TypedDict):
    """
    ReportStateInput クラス - レポート作成の初期入力を表現するデータ型
    
    レポート作成プロセスを開始するために必要な基本的な情報を定義します。
    """
    topic: str # レポートのトピック
    
class ReportStateOutput(TypedDict):
    """
    ReportStateOutput クラス - レポート作成の最終出力を表現するデータ型
    
    レポート作成プロセスの最終的な成果物を定義します。
    """
    final_report: str # 最終レポート

class ReportState(TypedDict):
    """
    ReportState クラス - レポート作成プロセス全体の状態を表現するデータ型
    
    レポート作成の各段階における状態情報を保持し、プロセス全体を追跡します。
    トピックから最終レポートまでの全情報を管理します。
    """
    topic: str # レポートのトピック    
    feedback_on_report_plan: str # レポート計画に対するフィードバック
    sections: list[Section] # レポートセクションのリスト 
    completed_sections: Annotated[list, operator.add] # Send() API用のキー
    report_sections_from_research: str # 最終セクションを書くための研究から完成したセクションの文字列
    final_report: str # 最終レポート

class SectionState(TypedDict):
    """
    SectionState クラス - 個別セクション作成プロセスの状態を表現するデータ型
    
    各セクションの作成における状態情報を保持します。
    検索プロセス、クエリ、取得した情報、および作成中のコンテンツを追跡します。
    """
    topic: str # レポートのトピック
    section: Section # レポートセクション  
    search_iterations: int # 実行した検索反復の回数
    search_queries: list[SearchQuery] # 検索クエリのリスト
    source_str: str # ウェブ検索から取得した整形済みソースコンテンツ
    report_sections_from_research: str # 最終セクションを書くための研究から完成したセクションの文字列
    completed_sections: list[Section] # 外部状態に複製する最終キー（Send() API用）

class SectionOutputState(TypedDict):
    """
    SectionOutputState クラス - セクション作成の出力状態を表現するデータ型
    
    セクション作成プロセスの結果を定義します。
    完成したセクションの集合を管理します。
    """
    completed_sections: list[Section] # 外部状態に複製する最終キー（Send() API用）
