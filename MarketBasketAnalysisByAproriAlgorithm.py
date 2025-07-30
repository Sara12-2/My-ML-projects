import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("🛒 Market Basket Analysis (Excel & CSV Supported)")

# Upload file (.csv or .xlsx)
uploaded_file = st.file_uploader("Upload your transaction file", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        # Read based on file type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")
        st.write("🧾 Available columns in your data:", df.columns)

        # Ensure required columns exist
        if 'InvoiceNo' not in df.columns or 'Description' not in df.columns:
            st.error("⚠️ Columns 'InvoiceNo' and 'Description' are required in the file!")
        else:
            # Clean data
            df.dropna(subset=["InvoiceNo", "Description"], inplace=True)
            df['InvoiceNo'] = df['InvoiceNo'].astype(str)

            # Group items by InvoiceNo
            transactions = df.groupby('InvoiceNo')['Description'].apply(list).reset_index(name='items')

            # ✅ Convert all items to string
            transactions['items'] = transactions['items'].apply(lambda x: [str(i) for i in x])

            # Encode transactions
            te = TransactionEncoder()
            te_array = te.fit(transactions['items']).transform(transactions['items'])
            df_encoded = pd.DataFrame(te_array, columns=te.columns_)

            # Support slider
            min_support = st.slider("Choose Minimum Support:", 0.01, 0.1, 0.02)

            # Run Apriori
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            st.subheader("📦 Frequent Itemsets")
            st.write(frequent_itemsets.sort_values(by="support", ascending=False))

            # Association rules
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            st.subheader("📈 Association Rules")
            st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by="lift", ascending=False))

            # Download button
            csv = rules.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Rules as CSV", csv, "association_rules.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
