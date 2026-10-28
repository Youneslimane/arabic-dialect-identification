"""Create the seven-page Arabic project report as a PDF."""

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Report_YOUNES_LIMANE.pdf"
FONT = Path(r"C:\Windows\Fonts\tahoma.ttf")
BOLD_FONT = Path(r"C:\Windows\Fonts\tahomabd.ttf")

pdfmetrics.registerFont(TTFont("Tahoma", str(FONT)))
pdfmetrics.registerFont(TTFont("Tahoma-Bold", str(BOLD_FONT)))


def rtl(text: str) -> str:
    return get_display(arabic_reshaper.reshape(text))


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ArabicBody",
        parent=styles["BodyText"],
        fontName="Tahoma",
        fontSize=9.2,
        leading=14,
        alignment=TA_RIGHT,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="ArabicSmall",
        parent=styles["ArabicBody"],
        fontSize=7.8,
        leading=11,
    )
)
styles.add(
    ParagraphStyle(
        name="ArabicHeading",
        parent=styles["Heading2"],
        fontName="Tahoma-Bold",
        fontSize=13,
        leading=17,
        alignment=TA_RIGHT,
        textColor=colors.HexColor("#183b56"),
        spaceBefore=4,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="ArabicTitle",
        parent=styles["Title"],
        fontName="Tahoma-Bold",
        fontSize=20,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#183b56"),
    )
)
styles.add(
    ParagraphStyle(
        name="English",
        parent=styles["ArabicBody"],
        fontName="Tahoma",
        alignment=TA_LEFT,
    )
)


def p(text: str, style: str = "ArabicBody") -> Paragraph:
    return Paragraph(rtl(text), styles[style])


def heading(text: str) -> Paragraph:
    return p(text, "ArabicHeading")


def table(rows: list[list[str]], widths: list[float]) -> Table:
    converted = [
        [Paragraph(rtl(cell), styles["ArabicSmall"]) for cell in row] for row in rows
    ]
    result = Table(converted, colWidths=widths, repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf2")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#91a9b5")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return result


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Tahoma", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=1.6 * cm,
    leftMargin=1.6 * cm,
    topMargin=1.45 * cm,
    bottomMargin=1.45 * cm,
    title="Arabic Dialect Identification",
    author="Younes Limane",
)
story: list = []

# Page 1: cover
story.extend(
    [
        Spacer(1, 3.2 * cm),
        p("مشروع معالجة اللغات الطبيعية", "ArabicTitle"),
        Spacer(1, 0.7 * cm),
        p("التعرف على اللهجات العربية على المستوى الإقليمي والقطري", "ArabicTitle"),
        Spacer(1, 1.2 * cm),
        p("باستخدام تقنيات التعلم الآلي", "ArabicTitle"),
        Spacer(1, 2.0 * cm),
        p("إعداد الطالب: Younes Limane", "ArabicBody"),
        p("السنة الدراسية: 2026", "ArabicBody"),
        Spacer(1, 4.0 * cm),
        p("تقرير مشروع NLP", "ArabicBody"),
        PageBreak(),
    ]
)

# Page 2: abstracts and introduction
story.extend(
    [
        heading("الملخص"),
        p(
            "يهدف هذا المشروع إلى بناء نظام آلي للتعرف على اللهجات العربية المكتوبة على "
            "مستويين: المنطقة والدولة. تم دمج بيانات IADD وMADAR بعد تنظيف النصوص "
            "وتوحيد التسميات وإزالة التكرار. استُخدم تمثيل Character TF-IDF مع نموذجي "
            "Logistic Regression وLinearSVC، ثم بُني نظام هرمي يحدد المنطقة أولًا ثم "
            "الدولة داخل المنطقة. حقق LinearSVC أفضل أداء إقليمي بدقة 87.57% وMacro-F1 "
            "بلغت 81.93%. كما طُورت واجهة Streamlit لإدخال النص وعرض النتيجة. ركزت "
            "النسخة الحالية على المنطقة والدولة، وتأجل تصنيف المدينة إلى عمل مستقبلي."
        ),
        heading("Abstract"),
        p(
            "This project presents a hierarchical system for identifying written Arabic "
            "dialects at the regional and country levels. IADD and MADAR corpora were "
            "cleaned, normalized, deduplicated, and split into training, validation, "
            "and test sets. Character TF-IDF was combined with Logistic Regression and "
            "LinearSVC classifiers. The system first predicts the region and then the "
            "country using a region-specific classifier. LinearSVC achieved 87.57% "
            "accuracy and 81.93% macro-F1 for regional classification. A Streamlit "
            "interface was developed for interactive testing. City-level classification "
            "is left for future work.",
            "English",
        ),
        p("الكلمات المفتاحية: معالجة اللغات الطبيعية، اللهجات العربية، IADD، MADAR، TF-IDF، LinearSVC."),
        heading("المقدمة العامة"),
        p(
            "تتميز اللغة العربية بتعدد لهجاتها واختلاف مفرداتها وتراكيبها حسب الموقع "
            "الجغرافي. لذلك فإن التعامل مع كل النصوص العربية على أنها فصحى أو لهجة "
            "واحدة يحد من دقة تطبيقات معالجة اللغات الطبيعية. تتمثل مشكلة المشروع في "
            "تحديد اللهجة من نص عربي مكتوب، مع تقسيم المهمة إلى مرحلتين: تحديد المنطقة "
            "ثم تحديد الدولة. تتمثل مساهمات المشروع في دمج مصادر مختلفة، وبناء نماذج "
            "أساسية قابلة للتشغيل، وتصميم تصنيف هرمي، وتوفير واجهة استخدام."
        ),
        heading("محتويات التقرير"),
        p("يتناول التقرير حالة الفن، ومصادر البيانات ومعالجتها، وتصميم النظام وتنفيذه، "
          "ثم التجارب والنتائج والخاتمة والمراجع."),
        PageBreak(),
    ]
)

# Page 3: state of art and data
story.extend(
    [
        heading("الفصل الأول: حالة الفن"),
        p(
            "تناولت أعمال سابقة مهمة التعرف على اللهجات العربية باستخدام نماذج التعلم "
            "الآلي والتعلم العميق. تعتمد بعض الأنظمة على التصنيف المباشر للدولة، بينما "
            "تستخدم أعمال أخرى بيانات مدينية أو نماذج Transformer عربية. يتميز هذا "
            "المشروع ببناء حل عملي منخفض التكلفة يعتمد على Character n-grams، مع فصل "
            "المشكلة إلى مستوى إقليمي ثم قطري. يسمح هذا التصميم بإضافة مستوى المدينة "
            "مستقبلًا."
        ),
        heading("الفصل الثاني: تحليل البيانات"),
        p("تم اختيار المصادر التالية وفق طبيعة التسميات وتوفر النصوص:"),
        table(
            [
                ["المصدر", "الاستخدام", "الملاحظة"],
                ["IADD", "التدريب", "نصوص عربية مع تسميات لهجية"],
                ["MADAR", "التدريب", "بيانات مرتبطة بمدن عربية"],
                ["NADI", "لم يستخدم في التدريب", "النسخة المتاحة TEST فقط"],
                ["QADI", "اختبار خارجي", "بيانات التدريب تعتمد على Twitter IDs"],
            ],
            [3.0 * cm, 5.2 * cm, 8.3 * cm],
        ),
        Spacer(1, 0.25 * cm),
        p("بعد المعالجة وإزالة التكرار أصبحت البيانات كما يلي:"),
        table(
            [
                ["المؤشر", "القيمة"],
                ["السجلات الخام", "247,804"],
                ["بعد المعالجة", "247,803"],
                ["بعد إزالة التكرار", "238,274"],
                ["IADD", "127,125"],
                ["MADAR", "111,149"],
            ],
            [8.8 * cm, 7.7 * cm],
        ),
        Spacer(1, 0.3 * cm),
        p("تم تقسيم البيانات إلى Train = 166,796، وValidation = 35,739، وTest = 35,739. "
          "لم تُستخدم سجلات Unknown في تدريب نموذج الدولة."),
        PageBreak(),
    ]
)

# Page 4: preprocessing and design
story.extend(
    [
        heading("المعالجة النصية"),
        p(
            "تختلف بنية ملفات المصادر، لذلك حُولت السجلات إلى صيغة موحدة تحتوي على: "
            "text، region، country، city، source. شملت المعالجة إزالة الروابط، والتشكيل، "
            "التطويل العربي، ومحارف التحكم، وتوحيد بعض الحروف العربية، وتوحيد الفراغات، "
            "ثم توحيد أسماء المناطق والدول وإزالة التكرار."
        ),
        p(
            "لم يتم استخدام Lemmatization أو Stopword Removal؛ لأن المصنف يعتمد على "
            "Character-level TF-IDF الذي يحافظ على المؤشرات الكتابية داخل الكلمات "
            "ويتناسب مع الاختلافات الإملائية واللهجية. كما لم تُختلق تسميات للمدن عند "
            "عدم توفرها في المصدر."
        ),
        heading("تصميم النظام"),
        p("التدفق العام للنظام هو:"),
        table(
            [
                ["المرحلة", "العملية"],
                ["1", "IADD + MADAR"],
                ["2", "تنظيف وتطبيع النصوص"],
                ["3", "توحيد التسميات وإزالة التكرار"],
                ["4", "تقسيم Train / Validation / Test"],
                ["5", "Character TF-IDF"],
                ["6", "LinearSVC لتحديد المنطقة"],
                ["7", "مصنف الدولة الخاص بالمنطقة"],
                ["8", "واجهة Streamlit"],
            ],
            [2.0 * cm, 14.5 * cm],
        ),
        Spacer(1, 0.3 * cm),
        heading("الاختيارات التقنية"),
        p(
            "اختير Character TF-IDF لأنه لا يتطلب محللًا صرفيًا، ويلتقط أجزاء الكلمات "
            "والأنماط الإملائية المميزة للهجات. استُخدم LinearSVC بسبب أدائه الجيد في "
            "التصنيف النصي عالي الأبعاد. أما التصنيف الهرمي فيقلل صعوبة التمييز بين "
            "عدد كبير من الدول دفعة واحدة."
        ),
        PageBreak(),
    ]
)

# Page 5: implementation
story.extend(
    [
        heading("الفصل الرابع: التنفيذ"),
        p("تم تنفيذ المشروع باستخدام Python 3.13 والمكتبات التالية: scikit-learn، "
          "joblib، matplotlib، وStreamlit. أُديرت الشفرة باستخدام Git، وحُفظت النماذج "
          "والنتائج في مجلدات مستقلة."),
        table(
            [
                ["الملف", "الوظيفة"],
                ["prepare_data.py", "قراءة وتنظيف ودمج وتقسيم البيانات"],
                ["train_baselines.py", "تدريب Logistic Regression وLinearSVC"],
                ["train_hierarchical.py", "تدريب نماذج الدول حسب المنطقة"],
                ["predict.py", "تنفيذ التنبؤ الهرمي لنص واحد"],
                ["app.py", "واجهة Streamlit التفاعلية"],
            ],
            [6.0 * cm, 9.5 * cm],
        ),
        Spacer(1, 0.35 * cm),
        heading("التدريب وحفظ النماذج"),
        p(
            "دُرب نموذجان أساسيان على مستوى المنطقة والدولة: Logistic Regression وLinearSVC. "
            "استُخدم Character TF-IDF بمدى n-grams من 3 إلى 5 وبحد أقصى 300,000 ميزة. "
            "حُفظت vectorizers والمصنفات بصيغة joblib، كما حُفظت المقاييس ومصفوفات "
            "الالتباس بصيغة JSON وPNG."
        ),
        heading("الصعوبات والحلول"),
        p(
            "تمثلت الصعوبات في اختلاف تنسيقات الملفات، ونقص بيانات التدريب في NADI، "
            "واعتماد QADI على معرفات Twitter بدل النصوص. عولجت هذه المشكلات بإنشاء "
            "قارئ موحد، وعدم استخدام TEST للتدريب، وحفظ QADI المتاح كاختبار خارجي. "
            "كما استُخدم التصنيف الهرمي لتقليل الالتباس بين اللهجات المتقاربة."
        ),
        heading("واجهة الاستخدام"),
        p(
            "تسمح الواجهة للمستخدم بإدخال نص عربي، ثم تعرض شريط تقدم أثناء المعالجة، "
            "وتقدم اسم المنطقة بالعربية واسم الدولة مع صورة العلم. الواجهة الحالية "
            "مخصصة للمنطقة والدولة، ولا تعرض المدينة."
        ),
        PageBreak(),
    ]
)

# Page 6: experiments and figures
story.extend(
    [
        heading("الفصل الخامس: التجارب والنتائج"),
        p("اعتمد التقييم على Accuracy وPrecision وRecall وMacro-F1 وWeighted-F1، مع "
          "استخدام مجموعة الاختبار التي لم تدخل في تدريب المصنف."),
        table(
            [
                ["المهمة", "النموذج", "Accuracy", "Macro-F1"],
                ["Region", "Logistic Regression", "83.54%", "76.58%"],
                ["Region", "LinearSVC", "87.57%", "81.93%"],
                ["Country", "Logistic Regression", "74.68%", "63.97%"],
                ["Country", "LinearSVC", "78.77%", "67.92%"],
            ],
            [3.0 * cm, 5.8 * cm, 3.2 * cm, 3.5 * cm],
        ),
        Spacer(1, 0.3 * cm),
        p("في التصنيف الهرمي بلغت النتائج: Levantine Accuracy = 81.14% وMacro-F1 = "
          "76.42%، وMaghrebi Accuracy = 91.08% وMacro-F1 = 88.43%، وGulf Accuracy = "
          "74.32% وMacro-F1 = 63.67%. تستخدم المصرية والعراقية تعيينًا ثابتًا لأن كل "
          "منطقة تحتوي على دولة واحدة في البيانات المستخدمة."),
        Spacer(1, 0.2 * cm),
        Image(str(ROOT / "results" / "region_linear_svc_confusion_matrix.png"), width=8.4 * cm, height=6.0 * cm),
        p("الشكل 1: مصفوفة الالتباس لتصنيف المنطقة باستخدام LinearSVC.", "ArabicSmall"),
        heading("مناقشة النتائج والقيود"),
        p(
            "تفوق LinearSVC على Logistic Regression في المهمتين. يرجع انخفاض أداء الدولة "
            "إلى تشابه اللهجات داخل المنطقة وعدم توازن عدد الأمثلة. كما أن خطأ تحديد "
            "المنطقة ينتقل إلى المرحلة الثانية في النظام الهرمي. لذلك ينبغي تفسير "
            "الناتج كتنبؤ احتمالي، خصوصًا مع النصوص القصيرة جدًا."
        ),
        PageBreak(),
    ]
)

# Page 7: conclusion, references, appendix
story.extend(
    [
        heading("الخاتمة والآفاق"),
        p(
            "تم بناء نظام عملي للتعرف على اللهجات العربية على مستوى المنطقة والدولة. "
            "دمج النظام بيانات IADD وMADAR، وطبق تنظيفًا وتوحيدًا للتسميات، ثم استخدم "
            "Character TF-IDF مع LinearSVC داخل بنية هرمية. حقق النظام 87.57% Accuracy "
            "على مستوى المنطقة، كما أُنشئت واجهة Streamlit قابلة للتجربة. لم يُفعل "
            "تصنيف المدينة في النسخة الحالية."
        ),
        p(
            "تشمل التحسينات المستقبلية إضافة بيانات NADI وQADI عند توفر نصوص تدريب "
            "مرخصة، وتجربة AraBERT أو CAMeL-BERT، ومعالجة عدم توازن الفئات، وتفعيل "
            "تصنيف المدينة، ونشر الواجهة كخدمة ويب أو API، مع مراعاة الخصوصية وحقوق "
            "استخدام البيانات."
        ),
        heading("المراجع"),
        p("1. IADD Dataset, Hugging Face: https://huggingface.co/datasets/evageon/IADD"),
        p("2. MADAR Corpus: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/"),
        p("3. NADI Shared Task: https://github.com/UBC-NLP/nadi"),
        p("4. QADI Dataset: https://github.com/qcri/QADI"),
        p("5. Scikit-learn Documentation: https://scikit-learn.org/"),
        p("6. Streamlit Documentation: https://docs.streamlit.io/"),
        heading("الملحق: مثال تشغيل"),
        p('الأمر: python scripts\\predict.py --text "شو أخبارك اليوم؟"'),
        p("الناتج المتوقع تقريبًا: المنطقة الشامية، والدولة لبنان. النتائج تقديرية وقد "
          "تختلف للنصوص القصيرة أو المتشابهة."),
        heading("ملاحظة التسليم"),
        p("يُرفق هذا التقرير باسم Report_YOUNES_LIMANE.pdf مع ملف المصدر المضغوط "
          "Source_YOUNES_LIMANE.zip عبر نموذج الأستاذ قبل 1 ديسمبر 2026."),
    ]
)

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
