# Arabic Dialect Identification

نظام هرمي للتعرف على اللهجات العربية المكتوبة، يبدأ بتحديد المنطقة ثم يحدد
الدولة داخل المنطقة. تركز النسخة الحالية على **Region** و **Country**، بينما
تم تأجيل تصنيف المدينة إلى مرحلة لاحقة.

## فكرة المشروع

تختلف اللهجات العربية في المفردات والتهجئة والتراكيب حسب المنطقة والدولة.
يستخدم هذا المشروع Character TF-IDF مع LinearSVC لبناء مصنف عملي، ثم يطبق
التصنيف الهرمي:

```text
النص العربي -> المنطقة -> الدولة داخل المنطقة -> واجهة Streamlit
```

## البيانات

تم دمج مصدرين في التدريب:

- **IADD** من Hugging Face.
- **MADAR**، ويحتوي على بيانات مرتبطة بالمدن.

بعد التنظيف وإزالة التكرار:

- عدد السجلات: **238,274**
- IADD: **127,125**
- MADAR: **111,149**
- Train: **166,796**
- Validation: **35,739**
- Test: **35,739**

توجد ملفات NADI وQADI محليًا، لكن لم تُستخدم في التدريب الحالي: نسخة NADI
المتاحة هي TEST فقط، بينما بيانات QADI التدريبية تعتمد على Twitter IDs وتحتاج
إلى استرجاع النصوص بطريقة قانونية. حُفظ اختبار QADI المتاح في
`data/processed/qadi_test.jsonl`.

## النتائج

أفضل نموذج هو Character TF-IDF مع LinearSVC:

| المهمة | Accuracy | Macro-F1 |
|---|---:|---:|
| Region | 87.57% | 81.93% |
| Country المباشر | 78.77% | 67.92% |

أداء نماذج الدولة الهرمية يختلف حسب المنطقة؛ فقد بلغ تقريبًا:

- Levantine: Accuracy 81.14%، Macro-F1 76.42%.
- Maghrebi: Accuracy 91.08%، Macro-F1 88.43%.
- Gulf: Accuracy 74.32%، Macro-F1 63.67%.
- Egyptian وIraqi: تعيين ثابت لأن البيانات المتاحة تحتوي دولة واحدة لكل منطقة.

## بنية المشروع

```text
app.py                              واجهة Streamlit
scripts/prepare_data.py             تنظيف ودمج وتقسيم البيانات
scripts/train_baselines.py          تدريب Logistic Regression وLinearSVC
scripts/train_hierarchical.py       تدريب نماذج الدولة حسب المنطقة
scripts/predict.py                  التنبؤ الهرمي لنص واحد
data/processed/                     البيانات المنظفة والتقسيمات والتقرير
models/                             النماذج وTF-IDF المحفوظة
results/                            المقاييس ومصفوفات الالتباس
```

## التثبيت والتشغيل

يتطلب المشروع Python 3.10 أو أحدث:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

لتجهيز البيانات من الملفات المحلية:

```powershell
python scripts\prepare_data.py
```

لتدريب النماذج:

```powershell
python scripts\train_baselines.py
python scripts\train_hierarchical.py
```

للتنبؤ من سطر الأوامر:

```powershell
python scripts\predict.py --text "شو أخبارك اليوم؟"
```

لتشغيل الواجهة:

```powershell
python -m streamlit run app.py
```

## أمثلة

```text
شو أخبارك اليوم؟       -> الشامية / لبنان تقريبًا
إزيك عامل إيه؟         -> المصرية / مصر تقريبًا
واش خبارك اليوم؟       -> المغاربية / المغرب أو دولة مغاربية
```

النتائج تقديرية، وقد تتأثر بقصر النص أو تشابه اللهجات.

## حدود المشروع والعمل المستقبلي

- لم يتم تفعيل نموذج المدينة في الواجهة الحالية.
- يمكن تحسين النتائج بإضافة بيانات متوازنة واستعمال نماذج عربية Transformer.
- يمكن دمج NADI وQADI عند توفر نصوص تدريب مرخصة وقابلة للاستخدام.
- يمكن نشر الواجهة أو تحويلها إلى API.

## المصادر

- IADD: https://huggingface.co/datasets/evageon/IADD
- MADAR: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/
- NADI: https://github.com/UBC-NLP/nadi
- QADI: https://github.com/qcri/QADI
