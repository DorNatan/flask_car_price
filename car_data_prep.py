import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

def prepare_data(df):
    data = df.copy()
    
    try:
        # Dropping rows where 'Price' is NaN
        data = data.dropna(subset=['Price'])
    except Exception as e:
        print(f"Error dropping rows with NaN 'Price': {e}")
    
    try:
        # Dropping problematic text columns
        problematic_columns = ['Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Description', 'Color']
        data = data.drop(columns=[col for col in problematic_columns if col in data.columns])
    except Exception as e:
        print(f"Error dropping problematic text columns: {e}")
    
    try:
        # Handle missing values - filling missing numeric values with the mean
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    except Exception as e:
        print(f"Error filling missing numeric values: {e}")
    
    try:
        # Convert production year to car age
        current_year = 2024
        data['age'] = current_year - data['Year']
        data['age'] = data['age'].astype(np.int64)
    except Exception as e:
        print(f"Error converting 'Year' to 'age': {e}")
    
    try:
        # Correct spelling errors in the manufacturer column
        data['manufactor'] = data['manufactor'].replace({'Lexsus': 'לקסוס'})
    except Exception as e:
        print(f"Error correcting spelling errors in 'manufactor': {e}")
    
    try:
        # Remove unnecessary values in the model column
        def remove_values_from_model_column(df, column_name, values):
            for value in values:
                regex_pattern = re.escape(value)
                df[column_name] = df[column_name].str.replace(regex_pattern, '', regex=True, case=False)
            return df
        
        values_to_remove = [' / MITO', 'JUKE', 'הדור החדש', 'החדשה', '/ קבריולט', 'PHEV', '(\\(20..\\))', '\\r\\n ', '^\\s+|\\s+$', 'CIVIC', 'אונסיס']
        data = remove_values_from_model_column(data, 'model', values_to_remove)
    except Exception as e:
        print(f"Error removing values from 'model' column: {e}")
    
    try:
        # Create a new feature based on weights of existing features
        def create_weighted_feature(df, weights):
            normalized_df = df.copy()
            for column in weights.keys():
                if column in normalized_df.columns and np.issubdtype(df[column].dtype, np.number):
                    normalized_df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
            
            weighted_sum = sum(normalized_df[col] * weight for col, weight in weights.items() if col in normalized_df.columns and np.issubdtype(normalized_df[col].dtype, np.number))
            return weighted_sum
        
        weights = {'age': 0.2, 'Km': 0.3, 'Hand': 0.1, 'capacity_Engine': 0.4}
        data['Weighted_Feature'] = create_weighted_feature(data, weights)
    except Exception as e:
        print(f"Error creating 'Weighted_Feature': {e}")
    
    try:
        # One-Hot Encoding for categorical columns
        encoder = OneHotEncoder(sparse_output=False)
        
        # Combine manufacturer and model into one column
        data['manufactor_model'] = data['manufactor'] + ' ' + data['model']
        
        # Encode the columns
        for column in ['manufactor_model', 'Gear', 'Engine_type', 'ownership']:
            if column in data.columns:
                encoded = encoder.fit_transform(data[[column]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
                data = data.join(encoded_df)
    except Exception as e:
        print(f"Error during one-hot encoding: {e}")
    
    try:
        # Drop the original columns after encoding
        columns_to_drop = ['Year', 'Km', 'Hand', 'capacity_Engine', 'manufactor', 'model', 'Gear', 'Engine_type', 'ownership', 'manufactor_model']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    except Exception as e:
        print(f"Error dropping original columns after encoding: {e}")
    
    try:
        # Dropping any remaining non-numeric columns
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
        data = data.drop(columns=non_numeric_columns)
    except Exception as e:
        print(f"Error dropping non-numeric columns: {e}")
    
    return data
