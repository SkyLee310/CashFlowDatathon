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

category_linkage_clean = category_linkage_df.rename(columns={
    'Category Names': 'Category_Key',
    'Category': 'Flow_Type'
})

main_df = pd.merge(
    main_df,
    category_linkage_clean[['Category_Key', 'Flow_Type']],
    left_on='Category',
    right_on='Category_Key',
    how='left'
)

# 4. HANDLE "OTHER" CATEGORY
# Since 'Other' is not in the linkage file, we use logic:
# Negative values = Outflow, Positive = Inflow
main_df.loc[(main_df['Category'] == 'Other') & (main_df['Amount in USD'] < 0), 'Flow_Type'] = 'Outflow'
main_df.loc[(main_df['Category'] == 'Other') & (main_df['Amount in USD'] >= 0), 'Flow_Type'] = 'Inflow'

# 5. FINAL CLEANUP
# Remove the redundant 'Code' and 'Category_Key' columns created by the merge
main_df = main_df.drop(columns=['Code', 'Category_Key'])

# Verify results
print("Main DF Created Successfully!")
print(f"Total Transactions: {len(main_df)}")
print(main_df[['Pstng Date', 'Country', 'Flow_Type', 'Amount in USD']].head())

main_df.to_excel('Updated Dataset.xlsx', index=False)

