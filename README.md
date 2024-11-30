# UBS Entity Resolution

Developing Local Sensitive Hashing to find similiar non-UBS entities into the transaction records. 

## Preprocessing 

The preprocessing was made by lowercasing and deleting special characters.
For names, deletting common titles such as (dr., mrs., ...). 
For phone numbers, they were converted to only numbers. 

## Models
Local Sensitive Hashing (LSH) is utilized to reduce the number of entity comparisons required. Currently, LSH bucketing is performed based on names, ensuring that only entities within the same name bucket are compared.
