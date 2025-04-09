from pathlib import Path
import  os

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic
from typing import Hashable
from sdv.metadata import Metadata

from mcp.server.fastmcp import FastMCP
from pandas.core.interchange.dataframe_protocol import DataFrame
import pandas as pd

mcp = FastMCP("SDG-MCP-SERVER")

# Optional: set max columns and width to display full DataFrame
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Disable line wrapping


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


@mcp.tool()
def execute_python_script(script: str) -> str:
    """
    Executes a Python script provided as a string.

    Parameters:
        script (str): The Python code to be executed.

    Returns:
        str: "Success" if the script executes without errors.
             Otherwise, returns "Failure" followed by the error message.
    """
    try:
        exec(script)
        return "Success"
    except Exception as e:
        return f"Failure: {str(e)}"




@mcp.tool()
def generate_synthetic_data_from_gan(FilePath: str) -> str:
    """
    Generates synthetic tabular data using a CTGAN model trained on the input data.

    The function reads the dataset from the provided file path, trains a CTGAN model
    to learn the data distribution, generates synthetic samples, and saves them to a new
    file named 'Synthetic_{original_filename}.csv' in the same directory.

    Parameters:
        FilePath (str): The path to the CSV file containing the original data.

    Returns:Expected type 'Metadata', got 'SingleTableMetadata' instead
        str: "Success" if synthetic data was generated and saved successfully.
             Otherwise, returns "Failure" with an error message.
    """
    try:
        # Load data
        df = pd.read_csv(FilePath)
        #load meta
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)

        # Train CTGAN model

        model = CTGANSynthesizer(metadata=metadata)
        model.fit(df)

        # Generate synthetic data (same number of rows as original)
        synthetic_data = model.sample(len(df))

        # Create output filename
        dir_name, base_name = os.path.split(FilePath)
        synthetic_file_name = f"Synthetic_{base_name}"
        output_path = os.path.join(dir_name, synthetic_file_name)

        # Save synthetic data
        synthetic_data.to_csv(output_path, index=False)

        return "Success"
    except Exception as e:
        return f"Failure: {str(e)}"



if __name__=="__main__":
    mcp.run(transport="stdio")

