"""
Spreadsheet (.xlsx, .csv) Extraction
"""
import pandas as pd
from typing import Dict, List
from pathlib import Path

class SpreadsheetExtractor:
    """Extract data from Excel and CSV files"""

    def extract_excel(self, file_path: str) -> Dict:
        """Extract all sheets from Excel file"""
        try:
            # Read all sheets
            sheets = pd.read_excel(file_path, sheet_name=None)

            result = {
                "success": True,
                "sheet_count": len(sheets),
                "sheets": {}
            }

            for sheet_name, df in sheets.items():
                result["sheets"][sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data": df.to_dict(orient="records"),
                    "preview": df.head(5).to_string()
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def extract_csv(self, file_path: str) -> Dict:
        """Extract data from CSV file"""
        try:
            df = pd.read_csv(file_path)

            return {
                "success": True,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data": df.to_dict(orient="records"),
                "preview": df.head(10).to_string()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def format_for_llm(self, file_path: str) -> str:
        """Format spreadsheet for LLM consumption"""
        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            result = self.extract_csv(file_path)
            if not result["success"]:
                return ""

            output = [f"# CSV Data: {Path(file_path).name}\n"]
            output.append(f"**Rows:** {result['rows']}")
            output.append(f"**Columns:** {', '.join(result['column_names'])}\n")
            output.append("## Data Preview\n")
            output.append(result["preview"])

            return "\n".join(output)

        elif ext in [".xlsx", ".xls"]:
            result = self.extract_excel(file_path)
            if not result["success"]:
                return ""

            output = [f"# Excel File: {Path(file_path).name}\n"]
            output.append(f"**Sheets:** {result['sheet_count']}\n")

            for sheet_name, sheet_data in result["sheets"].items():
                output.append(f"## Sheet: {sheet_name}")
                output.append(f"**Rows:** {sheet_data['rows']}")
                output.append(f"**Columns:** {', '.join(sheet_data['column_names'])}\n")
                output.append("### Data Preview\n")
                output.append(sheet_data["preview"])
                output.append("")

            return "\n".join(output)

        return ""

# Test
if __name__ == "__main__":
    # Test extraction using the sample CSV
    extractor = SpreadsheetExtractor()
    result = extractor.extract_csv("docs/workshops/S7-sample-docs/employee_data.csv")

    print("=" * 60)
    print("CSV EXTRACTION TEST")
    print("=" * 60)
    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")
    print(f"Column Names: {result['column_names']}")
    print("\nLLM-Formatted:\n")
    print(extractor.format_for_llm("docs/workshops/S7-sample-docs/employee_data.csv"))