from pathlib import Path
import  os
from typing import Hashable

from mcp.server.fastmcp import FastMCP
from pandas.core.interchange.dataframe_protocol import DataFrame
import pandas as pd

mcp = FastMCP("SDG-MCP-SERVER")

@mcp.tool()
def add(a:int , b:int):
    """add two numbers"""
    return a + b

@mcp.tool()
def read_schema(FilePath : str) -> dict[Hashable, str]:
    """
    Reads a CSV file and returns its schema in dictionary format.

    Parameters:
        FilePath (str): The path to the CSV file.

    Returns:
        dict: A dictionary where each key is a column name and each value is the
              corresponding data type as a string (e.g., 'int64', 'float64', 'object').
    """
    df = pd.read_csv(FilePath)
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return schema



@mcp.tool()
def read_top_5_reccords(FilePath: str) -> pd.DataFrame:
    """
    Reads a CSV file from the specified path and returns the top 5 records.

    Parameters:
        FilePath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the first 5 rows of the file.
    """
    return pd.read_csv(FilePath).head(5)

if __name__=="__main__":
    mcp.run(transport="stdio")

