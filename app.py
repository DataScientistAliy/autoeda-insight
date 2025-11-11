# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import StringIO
import warnings

warnings.filterwarnings("ignore")

# Sahifa sozlamalari
st.set_page_config(page_title="AutoEDA Insight", layout="wide")
st.title("AutoEDA Insight")
st.caption("CSV yoki XLSX faylni yuklang — avtomatik tahlil qilamiz!")

# Fayl yuklash
uploaded_file = st.file_uploader("Faylni yuklang", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Faylni o'qish
        if uploaded_file.name.endswith('.csv'):
            # Encoding muammosini hal qilish uchun
            raw = uploaded_file.read().decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.success(f"Muvaffaqiyatli yuklandi! {df.shape[0]} qator, {df.shape[1]} ustun")

        # Sessiyada saqlash
        st.session_state.df = df.copy()
        st.session_state.df_original = df.copy()

    except Exception as e:
        st.error(f"Xatolik: {e}")
        st.stop()

    # === 1. Ma'lumot tozalash ===
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Tozalash oldin")
        st.write(df.head())

    # NaN, dublikat, type tuzatish
    df_clean = df.copy()

    # Dublikatlar
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        st.warning(f"{duplicates} ta dublikat o'chirildi")

    # NaN
    nan_cols = df_clean.isnull().sum()
    if nan_cols.sum() > 0:
        st.info(f"NaN qiymatlar soni: {nan_cols.sum()}")
        # Sonli ustunlar → median, kategorik → mode
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Noma'lum")

    # Column type tuzatish
    for col in df_clean.columns:
        # Sana aniqlash
        if 'date' in col.lower() or 'time' in col.lower() or 'sana' in col.lower():
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        # Sonli ko'rinishdagi matn
        elif df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col].str.replace(',', ''), errors='ignore')
            except:
                pass

    with col2:
        st.subheader("2. Tozalashdan keyin")
        st.write(df_clean.head())

    # Yangi df ni saqlash
    df = df_clean
    st.session_state.df_clean = df

    # === 2. Avtomatik grafikalar ===
    st.markdown("---")
    st.subheader("Avtomatik vizualizatsiyalar")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Umumiy", "Taqsimot", "Korrelyatsiya", "Vaqt seriyasi", "Kategorik"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Ma'lumot turlari**")
            st.write(df.dtypes.value_counts())
        with col2:
            st.write("**NaN (tozalangan)**")
            st.write(df.isnull().sum().sum())

    with tab2:
        if len(numeric_cols) > 0:
            fig = px.histogram(df, x=numeric_cols[0], title=f"{numeric_cols[0]} taqsimoti")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            df_ts = df.copy()
            df_ts['oy'] = df_ts[date_col].dt.to_period('M')
            monthly = df_ts.groupby('oy')[num_col].sum().reset_index()
            monthly['oy'] = monthly['oy'].astype(str)
            fig = px.line(monthly, x='oy', y=num_col, title=f"Oylik {num_col}")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            top10 = df[cat_col].value_counts().head(10)
            fig = px.bar(x=top10.index, y=top10.values, title=f"Eng ko'p {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    # === 3. Insight ===
    st.markdown("---")
    st.subheader("Yashirin trendlar (Insight)")

    insights = []

    # Insight 1: Eng katta sonli ustun
    if len(numeric_cols) > 0:
        max_col = df[numeric_cols].sum().idxmax()
        max_val = df[max_col].sum()
        insights.append(f"**Eng katta hajm:** `{max_col}` → {max_val:,.0f}")

    # Insight 2: O'sish (agar sana bo'lsa)
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        df_ins = df.copy()
        df_ins['oy'] = df_ins[date_col].dt.to_period('M')
        monthly = df_ins.groupby('oy')[num_col].sum()
        if len(monthly) > 1:
            growth = ((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100)
            insights.append(f"**O'sish:** So'nggi oyda `{num_col}` {growth:+.1f}% o'zgardi")

    for ins in insights:
        st.success(ins)