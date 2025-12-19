import pandas as pd
import openpyxl

main_excel = pd.read_excel("Datathon Dataset.xlsx")
main_excel['Pstng Date']=pd.to_datetime(main_excel['Pstng Date'])

main_excel.to_excel('Updated Dataset.xlsx', index=False)
print(main_excel['Pstng Date'])