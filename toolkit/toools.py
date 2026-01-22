from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lancedb.rerankers import LinearCombinationReranker
from langchain_community.vectorstores import LanceDB
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from data_models.models import RAGtoolSchema


from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper
from data_models.models import RAGtoolSchema


@tool(args_schema=RAGtoolSchema)
def retriever_tool(question: str):
    """This is a retriever tool"""
    return f"Retrieved documents for: {question}"


# Community tools (already tools)
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)

polygon_tool = PolygonFinancials(
    api_wrapper=PolygonAPIWrapper()
)

bing_tool = BingSearchResults()


def get_all_tools():
    return [
        retriever_tool,
        tavily_tool,
        polygon_tool,
        bing_tool
    ]


if __name__ == "__main__":
    tools = get_all_tools()
    for t in tools:
        print(t.name)
