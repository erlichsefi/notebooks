import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import dotenv

dotenv.load_dotenv(".env")
def extract_text_from_pdf(file):
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(file.getbuffer())

        text = ""
        with open(f.name, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        return text

def create_pdf_from_text(text):
    writer = PdfWriter()
    writer.add_page()
    writer.pages[0].append_text(text)
    output_pdf = BytesIO()
    writer.write(output_pdf)
    return output_pdf.getvalue()

def call_open_ai(prompt, model="gpt-3.5-turbo"):
    from openai import OpenAI

    client = OpenAI()

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return stream.choices[0].message.content

def main():
    st.title("CV optimater")

    uploaded_file = st.file_uploader("Upload your current CV", type="pdf")

    position = st.text_area("The poistion copied details:")

    prompt = st.text_area("The prompt",value="""
    Here is a user CV:
    {cv_text}

    and there a position he or she is intreasted in:
    {position}

    optimize the cv to the position, don't invent details.
    - provide the response in markdown format.
    - don't state the position company name in the response
    """)

    if st.button("Optimize my CV!"):
        cv_text = extract_text_from_pdf(uploaded_file)
        
        if cv_text.strip():
            word_count = call_open_ai(prompt.format(position=position,cv_text=cv_text))
            st.write(word_count)
        else:
            st.warning("Error when reading the CV")

if __name__ == "__main__":
    main()
