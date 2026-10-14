---
library_name: transformers
tags:
- pii-detection
- ner
- PII
- privacy
- token-classification
license: mit
datasets:
- nvidia/Nemotron-PII
language:
- en
metrics:
- f1
- precision
- recall
base_model:
- jhu-clsp/ettin-encoder-32m
model-index:
  - name: ettin-32m-nemotron-pii
    results:
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          name: nvidia/Nemotron-PII 
          type: nvidia/Nemotron-PII
          split: test (10K instances)
        metrics:
          - type: f1
            value: 0.9573
            name: F1 (micro)
          - type: precision
            value: 0.9596
            name: Precision
          - type: recall
            value: 0.9549
            name: Recall
---

# ettin-32m-nemotron-pii model

Light Weight PII Detection Model | Open Source | 32M Parameters | 95.73 F1 Score | [Blog Post](https://huggingface.co/blog/kalyan-ks/tiny-pii-entity-detection-models)

## Overview

Ettin-32m-nemotron-pii is based on the ettin-encoder-32M model and fine-tuned over the Nemotron PII dataset. This model can detect 50+ PII entities in both structured and unstructured texts across various domains like healthcare, finance, legal, cybersecurity etc. With just 32M parameters, the model achieves a strong F1-score of 95.73.

## Key Features 

- Achieves strong F1-score of 95.73 with just 32M parameters.
- Outperforms popular LLMs like DeepSeek-V4-Flash (84.89) and GPT-4o-Mini (78.69). 
- Detects 50+ PII entities in both structured and unstructured texts.
- Handles text in various domains like healthcare, finance, legal etc.

## Supported PII Entity Types

This model can detect the following 55 PII entity types

<details>
<summary> PII entity types with description </summary>

| Entity                              | Description                                      |
|-------------------------------------|--------------------------------------------------|
| account_number                      | Account Number                                   |
| age                                 | Age                                              |
| api_key                             | API Key                                          |
| bank_routing_number                 | Bank Routing Number                              |
| biometric_identifier                | Biometric Identifier                             |
| blood_type                          | Blood Type                                       |
| certificate_license_number          | Certificate or License Number                    |
| city                                | City                                             |
| company_name                        | Company Name                                     |
| coordinate                          | Geographic Coordinate                            |
| country                             | Country                                          |
| county                              | County                                           |
| credit_debit_card                   | Credit or Debit Card Number                      |
| customer_id                         | Customer ID                                      |
| cvv                                 | Card Verification Value (CVV)                    |
| date                                | Date                                             |
| date_of_birth                       | Date of Birth                                    |
| date_time                           | Date and Time                                    |
| device_identifier                   | Device Identifier                                |
| education_level                     | Education Level                                  |
| email                               | Email Address                                    |
| employee_id                         | Employee ID                                      |
| employment_status                   | Employment Status                                |
| fax_number                          | Fax Number                                       |
| first_name                          | First Name                                       |
| gender                              | Gender                                           |
| health_plan_beneficiary_number      | Health Plan Beneficiary Number                   |
| http_cookie                         | HTTP Cookie                                      |
| ipv4                                | IPv4 Address                                     |
| ipv6                                | IPv6 Address                                     |
| language                            | Language                                         |
| last_name                           | Last Name                                        |
| license_plate                       | Vehicle License Plate                            |
| mac_address                         | MAC Address                                      |
| medical_record_number               | Medical Record Number                            |
| national_id                         | National Identification Number                   |
| occupation                          | Occupation                                       |
| password                            | Password                                         |
| phone_number                        | Phone Number                                     |
| pin                                 | Personal Identification Number (PIN)             |
| political_view                      | Political View                                   |
| postcode                            | Postcode / Zip Code                              |
| race_ethnicity                      | Race or Ethnicity                                |
| religious_belief                    | Religious Belief                                 |
| sexuality                           | Sexuality / Sexual Orientation                   |
| ssn                                 | Social Security Number                           |
| state                               | State                                            |
| street_address                      | Street Address                                   |
| swift_bic                           | SWIFT / BIC Code                                 |
| tax_id                              | Tax Identification Number                        |
| time                                | Time                                             |
| unique_id                           | Unique Identifier                                |
| url                                 | URL / Web Address                                |
| user_name                           | Username                                         |
| vehicle_identifier                  | Vehicle Identification Number (VIN)              |

</details>

## Usage

```python

# First install Hugging Face transformers library
!pip install transformers

# Initialize and run the PII detection pipeline to extract PII entities
from transformers import pipeline

## Initialize the PII detection pipeline
ner = pipeline("ner", model="kalyan-ks/ettin-32m-nemotron-pii", aggregation_strategy="simple")

input_text = "Kalyan KS is from India. His email id is kalyan.ks@yahoo.com"

## Run the PII detection to extract PII entities
pii_entities = ner(input_text)

## Process the extracted PII entities 
def format_pii_entities(entities, original_text):
    if not entities:
        return []

    merged_entities = []

    entities = sorted(entities, key=lambda x: x['start'])

    current_entity = {
        'start': entities[0]['start'],
        'end': entities[0]['end'],
        'label': entities[0]['entity_group'],
        'text': entities[0]['word']
    }

    for next_ent in entities[1:]:
        is_same_label = next_ent['entity_group'] == current_entity['label']
        is_adjacent = next_ent['start'] <= current_entity['end'] + 1

        if is_same_label and is_adjacent:
            current_entity['end'] = max(current_entity['end'], next_ent['end'])
            current_entity['text'] = original_text[current_entity['start']:current_entity['end']]
        else:
            merged_entities.append(clean_entity(current_entity))
            current_entity = {
                'start': next_ent['start'],
                'end': next_ent['end'],
                'label': next_ent['entity_group'],
                'text': next_ent['word']
            }

    merged_entities.append(clean_entity(current_entity))
    return merged_entities

def clean_entity(ent):

    raw_text = ent['text']
    stripped_text = raw_text.strip()
    leading_spaces = len(raw_text) - len(raw_text.lstrip())

    return {
        'start': ent['start'] + leading_spaces,
        'end': ent['start'] + leading_spaces + len(stripped_text),
        'text': stripped_text,
        'label': ent['label']
    }

# Display the extracted PII entities
formatted_entities = format_pii_entities(pii_entities, input_text)
print(formatted_entities)

# Output
[{'start': 0, 'end': 9, 'text': 'Kalyan KS', 'label': 'first_name'}, {'start': 18, 'end': 23, 'text': 'India', 'label': 'country'}, {'start': 41, 'end': 60, 'text': 'kalyan.ks@yahoo.com', 'label': 'email'}]
```

## Evaluation

This model is evaluated on a 10k sample test set from Neomotron PII dataset and achieved the following results

| Metric    | Score  |
|-----------|--------|
| F1        | 95.73  |
| Precision | 95.96  |
| Recall    | 95.49  |
| Accuracy  | 99.17  |

### Top Performing PII Entity Types

| Entity                 | Precision | Recall | F1     |
|------------------------|-----------|--------|--------|
| mac_address            | 0.9965    | 1.0000 | 0.9982 |
| biometric_identifier   | 0.9957    | 0.9978 | 0.9967 |
| date_of_birth          | 0.9937    | 0.9976 | 0.9956 |
| email                  | 0.9937    | 0.9940 | 0.9939 |
| api_key                | 0.9921    | 0.9949 | 0.9935 |
| coordinate             | 0.9910    | 0.9955 | 0.9932 |
| vehicle_identifier     | 0.9863    | 0.9981 | 0.9922 |
| medical_record_number  | 0.9960    | 0.9881 | 0.9921 |
| employee_id            | 0.9909    | 0.9915 | 0.9912 |
| credit_debit_card      | 0.9935    | 0.9870 | 0.9902 |

### Challenging PII Entity Types

| Entity            | Precision | Recall | F1     |
|-------------------|-----------|--------|--------|
| occupation        | 0.7110    | 0.5115 | 0.5949 |
| time              | 0.8614    | 0.7784 | 0.8178 |
| political_view    | 0.8348    | 0.8842 | 0.8588 |
| age               | 0.8220    | 0.9184 | 0.8676 |
| state             | 0.8941    | 0.8570 | 0.8751 |
| national_id       | 0.8671    | 0.8999 | 0.8832 |
| company_name      | 0.8860    | 0.8860 | 0.8860 |
| fax_number        | 0.9070    | 0.8742 | 0.8903 |
| race_ethnicity    | 0.8611    | 0.9299 | 0.8942 |
| education_level   | 0.9232    | 0.8874 | 0.9049 |

## Limitations

- **Language**: This model works well only for English language texts.
- **Challenging PII Entity Types**: Some of the entity types like `occupation` has low F1 score.

## Citation

```bibtex
@misc{ettin-32m-pii-2026,
  title = {ettin-32m-nemotron-pii-2026: PII Detection Model},
  author = {Kalyan KS},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/kalyan-ks/ettin-32m-nemotron-pii}
}
```