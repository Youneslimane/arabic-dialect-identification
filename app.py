"""Streamlit interface for hierarchical Arabic dialect identification."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from predict import predict  # noqa: E402

REGION_ARABIC = {
    "Egyptian": "المصرية",
    "Gulf": "الخليجية",
    "Iraqi": "العراقية",
    "Levantine": "الشامية",
    "Maghrebi": "المغاربية",
    "MSA": "الفصحى",
}

COUNTRY_DISPLAY = {
    "Egypt": ("مصر", "eg"),
    "Lebanon": ("لبنان", "lb"),
    "Syria": ("سوريا", "sy"),
    "Jordan": ("الأردن", "jo"),
    "Palestine": ("فلسطين", "ps"),
    "Iraq": ("العراق", "iq"),
    "Algeria": ("الجزائر", "dz"),
    "Morocco": ("المغرب", "ma"),
    "Tunisia": ("تونس", "tn"),
    "Libya": ("ليبيا", "ly"),
    "Saudi Arabia": ("السعودية", "sa"),
    "Qatar": ("قطر", "qa"),
    "Oman": ("عُمان", "om"),
    "Yemen": ("اليمن", "ye"),
    "Sudan": ("السودان", "sd"),
}


st.set_page_config(
    page_title="Arabic Dialect Identifier",
    page_icon="🌐",
    layout="centered",
)

st.title("كاشف اللهجات العربية")
st.markdown("أدخل نصًا عربيًا للتعرف على المنطقة والدولة.")

text = st.text_area(
    "النص العربي",
    placeholder="مثال: شو أخبارك اليوم؟",
    height=150,
)

if st.button("تحليل اللهجة", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("يرجى إدخال نص عربي أولًا.")
    else:
        progress = st.progress(0, text="جارٍ تجهيز النص...")
        try:
            progress.progress(25, text="جارٍ تحميل النموذج...")
            progress.progress(60, text="جارٍ تحليل المنطقة...")
            result = predict(text)
            progress.progress(90, text="جارٍ تحديد الدولة...")
            progress.progress(100, text="اكتملت المعالجة")
        except (OSError, ValueError, RuntimeError) as error:
            progress.empty()
            st.error(f"تعذر إجراء التنبؤ: {error}")
        else:
            progress.empty()
            st.success("تم تحليل النص")
            first, second = st.columns(2)
            with first:
                st.metric(
                    "المنطقة المتوقعة",
                    REGION_ARABIC.get(result["region"], result["region"]),
                )
            with second:
                country_name, country_code = COUNTRY_DISPLAY.get(
                    result["country"],
                    (result["country"], None),
                )
                st.markdown("**الدولة المتوقعة**")
                country_name_column, country_flag_column = st.columns([3, 1])
                with country_name_column:
                    st.markdown(
                        f'<div dir="rtl" style="font-size: 1.8rem; '
                        f'font-weight: 600; text-align: right;">'
                        f'{country_name}</div>',
                        unsafe_allow_html=True,
                    )
                with country_flag_column:
                    if country_code:
                        st.image(
                            f"https://flagcdn.com/w80/{country_code}.png",
                            width=48,
                        )
                    else:
                        st.markdown("🌐")
            with st.expander("عرض النص بعد المعالجة"):
                st.write(result["text"])
