# Feature_engineering
import logging
import pandas as pd
import numpy as np

# Get logger
logger = logging.getLogger(__name__)

def add_features(df):
    """
    Add engineered features to the Advertising dataset.
    
    Features added:
    - Total_Ads: sum of TV + Radio + Newspaper
    - Interaction terms: TV*Radio, TV*Newspaper, Radio*Newspaper
    """
    try:
        df = df.copy()
        
        # Total spend
        df['Total_Ads'] = df['TV'] + df['Radio'] + df['Newspaper']
        
        # Interaction features
        df['TV_Radio'] = df['TV'] * df['Radio']
        df['TV_Newspaper'] = df['TV'] * df['Newspaper']
        df['Radio_Newspaper'] = df['Radio'] * df['Newspaper']

        logger.info(f" Feature engineering applied successfully. New columns added: "
                    f"{['Total_Ads', 'TV_Radio', 'TV_Newspaper', 'Radio_Newspaper']}")
        return df
    except Exception as e:
        logger.error(f" Error in feature engineering: {e}")
        raise
