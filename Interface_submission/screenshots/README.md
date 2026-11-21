# UI screenshots

Screenshots are taken from the UI actually running (`04_interface_demo.ipynb` on Google Colab, T4 GPU, fine-tuned model).

| File | Contents |
|---|---|
| `01_query_result.png` | Example **Online Store** database: a question answered as a sentence by the LLM, with the SQL available and a green guardrail badge |
| `02_guardrail_rejects.png` | A dangerous request ("delete all customers"): the guardrail rejects the `DELETE` (red badge) and nothing is executed |
| `03_uploaded_data.png` | **My data** mode: `sales.csv` (fictional data) uploaded, the schema filled in automatically, and questions answered from the data |
| `04_no_match_handling.png` | A question the data cannot answer: the app replies "No matching data was found." instead of crashing or inventing an answer |

All data in the images is fictional; no tokens or personal information are shown.
