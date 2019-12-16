Medical Cost Prediction Tool documentation

-CS410: Text Information Systems

Fall 2019

Ishan Babbar (team lead) [ibabbar2@illinois.edu](mailto:ibabbar2@illinois.edu)

Houmin Zhong [houminz2@illinois.edu](mailto:houminz2@illinois.edu)

**Please open**  **MCPT documentation.docx**  **for detail documentation**

## API setup (Windows 10)

1. Installing the requirements.txt - pip install –r requirements.txt
2. Run the Falcon API - python main.py
3. Run Postman (if you do not have it install, [https://www.getpostman.com](https://www.getpostman.com)) to make inferences on the model
4. Create a request
5. Select &quot;POST&quot;
6. Enter &quot;localhost:8080/invocations&quot; in request URL
7. Enter text, such as &quot;Heart Transplantation&quot;
8. Click &quot;Send&quot;
9. Prediction should appear in Response section
