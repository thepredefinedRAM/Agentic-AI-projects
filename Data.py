import requests

# Replace with your actual QuestDB IP and Port
questdb_ip = "40.81.240.69"  # No 'http://'
questdb_port = "9000"

# Function to send a SQL query to QuestDB
def run_query(query):
    url = f"http://{questdb_ip}:{questdb_port}/exec"
    response = requests.get(url, params={'query': query})
    if response.status_code != 200:
        raise Exception(f"Query failed: {response.text}")
    return response.json()

# === ELECTRONICS TABLE ===
print("🔌 Creating 'electronics' table...")
run_query("""
CREATE TABLE electronics (
    product_name STRING,
    price_in_dollars DOUBLE,
    count INT,
    expiry_date DATE,
    ts TIMESTAMP
) timestamp(ts);
""")
print("✅ 'electronics' table created!\n")

print("📝 Inserting electronics data...")
for q in [
    "INSERT INTO electronics VALUES ('Smartphone', 699.99, 30, '2026-12-31', '2025-04-18T10:00:00Z')",
    "INSERT INTO electronics VALUES ('Laptop', 1199.49, 20, '2027-06-30', '2025-04-18T10:01:00Z')",
    "INSERT INTO electronics VALUES ('Headphones', 199.99, 50, '2026-03-15', '2025-04-18T10:02:00Z')",
    "INSERT INTO electronics VALUES ('Smartwatch', 249.99, 40, '2026-09-01', '2025-04-18T10:03:00Z')",
    "INSERT INTO electronics VALUES ('Tablet', 329.00, 25, '2027-01-20', '2025-04-18T10:04:00Z')"
]:
    run_query(q)
print("✅ Electronics data inserted!\n")

# === FOOD TABLE ===
print("🍔 Creating 'food' table...")
run_query("""
CREATE TABLE food (
    product_name STRING,
    price_in_dollars DOUBLE,
    count INT,
    expiry_date DATE,
    ts TIMESTAMP
) timestamp(ts);
""")
print("✅ 'food' table created!\n")

print("📝 Inserting food data...")
for q in [
    "INSERT INTO food VALUES ('Bread', 2.99, 100, '2025-05-01', '2025-04-18T11:00:00Z')",
    "INSERT INTO food VALUES ('Milk', 1.99, 80, '2025-04-25', '2025-04-18T11:01:00Z')",
    "INSERT INTO food VALUES ('Eggs', 3.49, 60, '2025-05-05', '2025-04-18T11:02:00Z')",
    "INSERT INTO food VALUES ('Butter', 4.79, 40, '2025-05-20', '2025-04-18T11:03:00Z')",
    "INSERT INTO food VALUES ('Cheese', 5.99, 30, '2025-06-10', '2025-04-18T11:04:00Z')"
]:
    run_query(q)
print("✅ Food data inserted!\n")

# === VEGETABLES TABLE ===
print("🥦 Creating 'vegetables' table...")
run_query("""
CREATE TABLE vegetables (
    product_name STRING,
    price_in_dollars DOUBLE,
    count INT,
    expiry_date DATE,
    ts TIMESTAMP
) timestamp(ts);
""")
print("✅ 'vegetables' table created!\n")

print("📝 Inserting vegetables data...")
for q in [
    "INSERT INTO vegetables VALUES ('Tomato', 1.29, 150, '2025-04-28', '2025-04-18T12:00:00Z')",
    "INSERT INTO vegetables VALUES ('Carrot', 0.99, 200, '2025-05-02', '2025-04-18T12:01:00Z')",
    "INSERT INTO vegetables VALUES ('Broccoli', 1.89, 120, '2025-04-30', '2025-04-18T12:02:00Z')",
    "INSERT INTO vegetables VALUES ('Spinach', 2.49, 90, '2025-04-26', '2025-04-18T12:03:00Z')",
    "INSERT INTO vegetables VALUES ('Potato', 0.79, 300, '2025-06-15', '2025-04-18T12:04:00Z')"
]:
    run_query(q)
print("✅ Vegetables data inserted!\n")

# === OPTIONAL: PREVIEW TABLES ===
print("🔍 Previewing 'electronics':")
print(run_query("SELECT * FROM electronics"))

print("🔍 Previewing 'food':")
print(run_query("SELECT * FROM food"))

print("🔍 Previewing 'vegetables':")
print(run_query("SELECT * FROM vegetables"))
