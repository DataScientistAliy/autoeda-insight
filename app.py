# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# PDF uchun
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# Sahifa sozlamalari
st.set_page_config(page_title="AutoEDA Insight", layout="wide")
sns.set_style("whitegrid")

st.title("AutoEDA Insight")
st.caption("CSV yoki XLSX faylni yuklang — avtomatik tahlil + PDF hisobot!")

# Fayl yuklash
uploaded_file = st.file_uploader("Faylni yuklang", type=["csv", "xlsx"])

# Excel sheet tanlagich
sheet_name = None
df = None

if uploaded_file is not None:
    try:
        # Fayl nomini tekshirish
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.success(f"CSV yuklandi: {df.shape[0]} qator, {df.shape[1]} ustun")
        else:
            # XLSX → sheet nomlarini olish
            excel_file = pd.ExcelFile(uploaded_file, engine='openpyxl')
            sheet_names = excel_file.sheet_names
            st.success(f"XLSX yuklandi: {len(sheet_names)} ta sheet topildi")

            # Sheet tanlagich
            sheet_name = st.selectbox("Sheet tanlang:", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
            st.info(f"**{sheet_name}** sheeti yuklandi: {df.shape[0]} qator, {df.shape[1]} ustun")

        st.session_state.df = df.copy()

    except Exception as e:
        st.error(f"Xatolik: {e}")
        st.stop()

    # === Tozalash ===
    df_clean = df.copy()
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates()
        st.warning(f"{duplicates} ta dublikat o'chirildi")

    nan_count = df_clean.isnull().sum().sum()
    if nan_count > 0:
        st.info(f"NaN qiymatlar: {nan_count}")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                mode_val = df_clean[col].mode()
                df_clean[col].fillna(mode_val[0] if not mode_val.empty else "Noma'lum", inplace=True)

    for col in df_clean.columns:
        if any(k in col.lower() for k in ['date', 'time', 'sana', 'year', 'month']):
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    df = df_clean

    # === Grafikalar ===
    st.markdown("---")
    st.subheader("Avtomatik vizualizatsiyalar")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    figures = []
    insights = []

    # 1. Histogram
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=ax, color='skyblue')
        ax.set_title(f"{numeric_cols[0]} taqsimoti")
        st.pyplot(fig)
        figures.append(fig)

    # 2. Korrelyatsiya
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='RdBu', center=0, ax=ax, fmt=".2f")
        ax.set_title("Korrelyatsiya matritsasi")
        st.pyplot(fig)
        figures.append(fig)

    # 3. Vaqt seriyasi
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        df_ts = df.copy()
        df_ts = df_ts.set_index(date_cols[0])[numeric_cols[0]].resample('M').sum()
        df_ts.plot(kind='line', marker='o', ax=ax, color='green')
        ax.set_title(f"Oylik {numeric_cols[0]} o'zgarishi")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        figures.append(fig)

    # 4. Kategorik
    if len(categorical_cols) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        top10 = df[categorical_cols[0]].value_counts().head(10)
        sns.barplot(x=top10.values, y=top10.index, ax=ax, palette="viridis")
        ax.set_title(f"Eng ko'p uchraydigan {categorical_cols[0]}")
        st.pyplot(fig)
        figures.append(fig)

    # === Insight ===
    st.markdown("---")
    st.subheader("Yashirin trendlar (Insight)")

    if len(numeric_cols) > 0:
        max_col = df[numeric_cols].sum().idxmax()
        max_val = df[max_col].sum()
        insights.append(f"**Eng katta hajm:** `{max_col}` → {max_val:,.0f}")

    if len(date_cols) > 0 and len(numeric_cols) > 0:
        monthly = df.set_index(date_cols[0])[numeric_cols[0]].resample('M').sum()
        if len(monthly) > 1:
            growth = ((monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0]) * 100
            insights.append(f"**O'sish:** `{numeric_cols[0]}` {growth:+.1f}% (birinchi va oxirgi oy)")

    for ins in insights:
        st.success(ins)

    # === PDF Hisobot ===
    st.markdown("---")
    if st.button("PDF Hisobotni yuklab olish", type="primary"):
        with st.spinner("PDF tayyorlanmoqda..."):
            pdf_buffer = generate_pdf_report(df, insights, figures, sheet_name)
            file_name = f"AutoEDA_{sheet_name or 'CSV'}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="Hisobotni yuklab olish (PDF)",
                data=pdf_buffer,
                file_name=file_name,
                mime="application/pdf"
            )
        st.success("PDF tayyor!")

# PDF funksiyasi (sheet nomi qo'shildi)
def generate_pdf_report(df, insights, figures, sheet_name=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # Sarlavha
    title = f"AutoEDA Insight Hisoboti — {sheet_name or 'CSV'}"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Yaratilgan: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Jadval
    story.append(Paragraph("Ma'lumotlar namunasi (birinchi 10 qator):", styles['Heading3']))
    data = [df.columns.tolist()] + df.head(10).values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Insight
    story.append(Paragraph("Yashirin trendlar:", styles['Heading3']))
    for ins in insights:
        story.append(Paragraph(f"• {ins.replace('**','').replace('`','')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Grafikalar
    story.append(Paragraph("Vizualizatsiyalar:", styles['Heading3']))
    for fig in figures:
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=6*inch, height=4*inch))
        story.append(Spacer(1, 12))
        plt.close(fig)

    doc.build(story)
    buffer.seek(0)
    return buffer
