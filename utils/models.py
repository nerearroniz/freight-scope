
def calculate_ftg_gonzalez_feliu_2020(sector, n_emp):
    # Calculate FTG
    ## Gonzalez-Feliu, 2020

    included_sectors = ['Agriculture','Craftsmen', 'Offices', 'Tertiary (non-offices) and services',
                        'Chemical industry', 'Construction industry', 'Primary and intermediate products',
                        'Food and non-fragile consumer goods', 'Wholesale', 'Department stores', 'Clothing, shoes, leather',
                        "Butcher's shops", 'Small groceries', 'Bakery retailers', 'Hotels, restaurants, cafés',
                        'Pharmacies', 'Hardware stores', 'Furnishing shops', 'Bookshops', 'Street trading (marketplaces)',
                        'Other retail shops', 'Only Transport', 'Transport and warehousing']

    if sector in included_sectors:

        model_dict = {'Agriculture': lambda E_num: 2.8 + (E_num * 0.21),
                      'Craftsmen': lambda E_num: 3.19 + (E_num * 1.01), 
                          'Offices': lambda E_num: 3.57 + (E_num * 0.02), 
                          'Tertiary (non-offices) and services': lambda E_num: 4.63,
                          'Chemical industry': lambda E_num: 23.88 + (E_num * 0.15), 
                          'Construction industry': lambda E_num: 6.57 + (E_num * 0.21), 
                          'Primary and intermediate products': lambda E_num: 6.52 + (E_num * 0.25),
                          'Food and non-fragile consumer goods': lambda E_num: 8.68 + (E_num * 0.23), 
                          'Wholesale': lambda E_num: 18.74 + (E_num * 0.68), 
                          'Department stores': lambda E_num: (E_num * 0.54), 
                          'Clothing, shoes, leather': lambda E_num: 2.01 + (E_num * 0.17),
                          "Butcher's shops": lambda E_num: 3.55 + (E_num * 1.18), 
                          'Small groceries': lambda E_num: 4.34 + (E_num * 1.02), 
                          'Bakery retailers': lambda E_num: 7.31, 
                          'Hotels, restaurants, cafés': lambda E_num: 2.63 + (E_num * 0.61),
                          'Pharmacies': lambda E_num: 15.94 + (E_num * 1.94), 
                          'Hardware stores': lambda E_num: 2.1 + (E_num * 0.87), 
                          'Furnishing shops': lambda E_num: 6.11 + (E_num * 0.1), 
                          'Bookshops': lambda E_num: 10.25, 
                          'Street trading (marketplaces)': lambda E_num: 5.77,
                          'Other retail shops': lambda E_num: (E_num * 0.96), 
                          'Only Transport': lambda E_num: 10.76, 
                          'Transport and warehousing': lambda E_num: (E_num * 0.04)}


        # Calculate FTG
        FTG = model_dict[sector](n_emp)

        return FTG
        
    else:
        return None
    
def apply_establishment_demand_model(df, sector_col, employee_col, model = "Sector (Gonzalez-Feliu et al, 2020)"):
        if model == "Sector (Gonzalez-Feliu et al, 2020)":
            # Calculate FTG for the establishments
            df.loc[:, 'ftg'] = df.apply(lambda row: calculate_ftg_gonzalez_feliu_2020(row[sector_col], row[employee_col]), axis=1)
            
        return df