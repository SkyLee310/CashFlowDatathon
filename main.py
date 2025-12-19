import pandas as pd
import openpyxl

file_path="Datathon Dataset.xlsx"
main_df = pd.read_excel(file_path, sheet_name='Data - Main')
cash_balance_df = pd.read_excel(file_path, sheet_name='Data - Cash Balance')
country_mapping_df = pd.read_excel(file_path, sheet_name='Others - Country Mapping')
category_linkage_df = pd.read_excel(file_path, sheet_name='Others - Category Linkage')
exchange_rate_df = pd.read_excel(file_path, sheet_name='Others - Exchange Rate')

main_df['Pstng Date']=pd.to_datetime(main_df['Pstng Date'])

main_df = pd.merge(
    main_df,
    country_mapping_df[['Code', 'Country', 'Currency']],
    left_on='Name',
    right_on='Code',
    how='left'
)

main_df.to_excel('Updated Dataset.xlsx', index=False)
print(main_df['Pstng Date'])
