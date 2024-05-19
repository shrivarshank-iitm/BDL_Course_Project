from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, udf, length, expr
from pyspark.sql.types import StringType, FloatType
import numpy as np

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Laptop Price Preprocessing") \
    .getOrCreate()

# Load the dataset
df = spark.read.csv('/kaggle/input/laptop-price-dataset-april-2024/raw_ebay.csv', header=True, inferSchema=True)

# Show initial schema and data
df.printSchema()
df.show()

# Function to replace null values with 'unknown'
def replace_null_with_unknown(df, columns):
    for column in columns:
        df = df.withColumn(column, when(col(column).isNull(), 'unknown').otherwise(col(column)))
    return df

# Columns to replace nulls with 'unknown'
columns_to_replace = ['Brand', 'RAM', 'Processor', 'GPU', 'GPU_Type', 'Resolution']
df = replace_null_with_unknown(df, columns_to_replace)

# Replace nulls in 'Product_Description' with 0
df = df.withColumn('Product_Description', when(col('Product_Description').isNull(), 0).otherwise(col('Product_Description')))

# Handle 'Screen_Size' column
def clean_screen_size(value):
    if value is None or any(x in str(value) for x in ['Does', 'N\\A', 'Not', 'Unknown']):
        return np.nan
    try:
        return float(regexp_replace(value, '[^0-9.]', ''))
    except:
        return np.nan

clean_screen_size_udf = udf(clean_screen_size, FloatType())
df = df.withColumn('Screen_Size', clean_screen_size_udf(col('Screen_Size')))

# Compute mean screen size and replace NaNs with mean
mean_screen_size = df.select('Screen_Size').agg({"Screen_Size": "mean"}).collect()[0][0]
df = df.withColumn('Screen_Size', when(col('Screen_Size').isNull(), mean_screen_size).otherwise(col('Screen_Size')))

# Drop rows where 'Price' is null
df = df.dropna(subset=['Price'])

# Convert 'Product_Description' to its length
df = df.withColumn('Product_Description', when(col('Product_Description').cast(StringType()).isNotNull(), 
                                               length(col('Product_Description'))).otherwise(col('Product_Description')))

# Standardize 'Brand' names
brand_replacements = {
    'Lenovo': 'lenovo', 'Dell': 'dell', 'ASUS': 'asus', 'HP': 'hp', 'Acer': 'acer', 'Microsoft': 'microsoft', 
    'Razer': 'razer', 'MSI': 'msi', 'Apple': 'apple', 'Samsung': 'samsung', 'Panasonic': 'panasonic', 
    'LG': 'lg', 'Geo': 'geo', 'DELL': 'dell', 'LENOVO': 'lenovo', 'Gateway': 'gateway', 'LG Electronics': 'lg', 
    'Huawei': 'huawei', 'Getac': 'getac', 'MICROSOFT': 'microsoft', 'Google': 'google', 'Dell Inc.': 'dell', 
    'Asus': 'asus', 'ThinkPad': 'thinkpad', 'Chuwi': 'chuwi', 'Sony': 'sony', 'Unbranded': 'unknown', 
    'VAIO': 'vaio', 'ByteSpeed': 'bytespeed', 'Dell gaming games game': 'dell', 'Eurocom': 'eurocom', 
    'Sager': 'sager', 'GIGABYTE': 'gigabyte', 'Alienware': 'alienware', 'AVITA': 'avita', 'Dell Latitude': 'dell', 
    'HP Commercial Remarketing': 'hp', 'Dell Commercial': 'dell', 'Lenovo Idea': 'lenovo', 'Microsoft Surface': 'microsoft', 
    'SAMSUNG': 'samsung'
}

brand_replacement_expr = " ".join([f"WHEN Brand = '{k}' THEN '{v}'" for k, v in brand_replacements.items()])
df = df.withColumn('Brand', expr(f"CASE {brand_replacement_expr} ELSE Brand END"))

# Clean up 'RAM' column
ram_replacements = {'Up': 'unknown', '16gb': '16', '8gb': '8', '16GB': '16', '8GB': '8', '4GB': '4', 
                    '32gb': '32', '32GB': '32', '4GB': '4', 'up': 'unknown', '8GB': '8', '64gb': '64', 
                    'upto': 'unknown', '16GB,': '16', '4GB,': '4', '8GB,': '8'}

ram_replacement_expr = " ".join([f"WHEN RAM = '{k}' THEN '{v}'" for k, v in ram_replacements.items()])
df = df.withColumn('RAM', expr(f"CASE {ram_replacement_expr} ELSE RAM END"))

# Convert to numeric columns where needed
numeric_columns = ['Processor', 'GPU', 'Resolution', 'Condition', 'GPU_Type', 'RAM', 'Brand']
for column in numeric_columns:
    df = df.withColumn(column, regexp_replace(col(column), '[^0-9.]', '').cast(FloatType()))

# Show final schema and data
df.printSchema()
df.show()

# Save the preprocessed DataFrame to a CSV file
output_path = "/kaggle/working/new_preprocessed_laptop_data.csv"
df.coalesce(1).write.csv(output_path, header=True)

print(f"Preprocessed data saved to {output_path}")