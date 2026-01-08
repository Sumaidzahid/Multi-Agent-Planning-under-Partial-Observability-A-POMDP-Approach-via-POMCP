import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path
        self.df = None
    # -------------------------------------------------------
    # 1. Loading Data from Excel
    # -------------------------------------------------------
    def load_data(self):
        try:
            # Attempt to read the excel file
            self.df = pd.read_excel(self.path)
            #print(self.df.describe())
            return self.df
        except FileNotFoundError:
            print(f"Error: The file '{self.path}' was not found in the current directory.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
        except pd.errors.ParserError:
            print("Error: Unable to parse the file. Please check if it's properly formatted.")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    loader = DataLoader('MAP_KRR.xlsx')
    data = loader.load_data()
    if data is not None:
        print("Data loaded successfully.")
