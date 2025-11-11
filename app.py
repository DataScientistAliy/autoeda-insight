# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# PDF uchun
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64

# Sahifa sozlamalari
st.set_page_config(page_title="AutoEDA Insight", layout="wide")
st.title("AutoEDA Insight")
st.caption("CSV yoki XLSX faylni yuklang — avtomatik tahlil + PDF hisobot!")

# Fayl yuklash
uploaded_file = st.file_uploader("Faylni yuklang", type=["csv", "xlsx"])

def generate_pdf_report(df, insights, figures):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []

    # Sarlavha
    story.append(Paragraph("AutoEDA Insight Hisoboti", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Yaratilgan sana: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Jadval (birinchi 10 qator)
    story.append(Paragraph("Ma'lumotlar namunasi (birinchi 10 qator):", styles['Heading3']))
    data = [df.columns.tolist()] + df.head(10).values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Insightlar
    story.append(Paragraph("Yashirin trendlar (Insight):", styles['Heading3']))
    for ins in insights:
        story.append(Paragraph(f"• {ins.replace('**','').replace('`','')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Grafikalar
    story.append(Paragraph("Vizualizatsiyalar:", styles['Heading3']))
    for i, fig in enumerate(figures):
        img_buffer = BytesIO()
        # Plotly → PNG
        fig.write_image(img_buffer, format="png", width=800, height=500, scale=2)
        img_buffer.seek(0)
        story.append(Image(img_buffer, width=6*inch, height=3.75*inch))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        st.success(f"Muvaffaqiyatli yuklandi! {df.shape[0]} qator, {df.shape[1]} ustun")
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

    nan_cols = df_clean.isnull().sum()
    if nan_cols.sum() > 0:
        st.info(f"NaN qiymatlar soni: {nan_cols.sum()}")
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Noma'lum")

    for col in df_clean.columns:
        if any(k in col.lower() for k in ['date', 'time', 'sana']):
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        elif df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col].str.replace(',', ''), errors='ignore')
            except:
                pass

    df = df_clean
    st.session_state.df_clean = df

    # === Grafikalar ===
    st.markdown("---")
    st.subheader("Avtomatik vizualizatsiyalar")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    figures = []
    insights = []

    # 1. Taqsimot
    if len(numeric_cols) > 0:
        fig1 = px.histogram(df, x=numeric_cols[0], title=f"{numeric_cols[0]} taqsimoti")
        st.plotly_chart(fig1, use_container_width=True)
        figures.append(fig1)

    # 2. Korrelyatsiya
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig2 = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu', title="Korrelyatsiya matritsasi")
        st.plotly_chart(fig2, use_container_width=True)
        figures.append(fig2)

    # 3. Vaqt seriyasi
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        df_ts = df.copy()
        df_ts['oy'] = df_ts[date_col].dt.to_period('M')
        monthly = df_ts.groupby('oy')[num_col].sum().reset_index()
        monthly['oy'] = monthly['oy'].astype(str)
        fig3 = px.line(monthly, x='oy', y=num_col, title=f"Oylik {num_col}")
        st.plotly_chart(fig3, use_container_width=True)
        figures.append(fig3)

    # 4. Kategorik
    if len(categorical_cols) > 0:
        cat_col = categorical_cols[0]
        top10 = df[cat_col].value_counts().head(10)
        fig4 = px.bar(x=top10.index, y=top10.values, title=f"Eng ko'p {cat_col}")
        st.plotly_chart(fig4, use_container_width=True)
        figures.append(fig4)

    # === Insight ===
    st.markdown("---")
    st.subheader("Yashirin trendlar (Insight)")

    if len(numeric_cols) > 0:
        max_col = df[numeric_cols].sum().idxmax()
        max_val = df[max_col].sum()
        insights.append(f"**Eng katta hajm:** `{max_col}` → {max_val:,.0f}")

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

    # === PDF Eksport ===
    st.markdown("---")
    if st.button("PDF Hisobotni yuklab olish", type="primary"):
        with st.spinner("PDF yaratilmoqda..."):
            pdf_buffer = generate_pdf_report(df, insights, figures)
            st.download_button(
                label="Hisobotni yuklab olish (PDF)",
                data=pdf_buffer,
                file_name=f"AutoEDA_Hisobot_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
        st.success("PDF tayyor!")
