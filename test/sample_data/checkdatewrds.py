import wrds

db = wrds.Connection()

print("Checking the database for available date range...")

check_sql = """
SELECT 
	MIN(date) AS first_date,
	MAX(date) AS last_date
FROM crsp.msf;
"""
df_date = db.raw_sql(check_sql)

first_available_date = df_date.iloc[0]["first_date"]
last_available_date = df_date.iloc[0]["last_date"]

print(f"WRDS (CRSP) earliest available date: {first_available_date}")
print(f"WRDS (CRSP) latest available date: {last_available_date}")

db.close()