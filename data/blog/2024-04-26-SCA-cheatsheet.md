---
layout: post # must be 'post'
title: 'Software Composition Analysis Tool Cheat Sheet'
subtitle: 'Guide to Efficiently Identifying Vulnerabilities in Software Components'
tags:
  - software-composition-analysis
  - cybersecurity
---

# Background

Identify corresponding CVE / Vulnerabilities in given python libraries and JavaScript `package.json`

# Tool Scan

## Python Pip SCA

1. Get the `requirements.txt` ready

2. Run

   ```shell
   cat requirements.txt | safety check --stdin --detailed-output json >test.json
   ```

3. Run

   ```python
   import json
   import csv
   with open('test.json', 'r') as file:
           data = json.load(file)
       with  open('python-report.csv', 'w', newline='') as csvfile:
           csv_writer = csv.writer(csvfile)
           csv_writer.writerow(['Component Name', 'Component Version', 'Vulnerability Description', 'CVE ID'])
           for data in data["vulnerabilities"]:
                csv_writer.writerow([data["package_name"],data["analyzed_version"],data["advisory"],data["CVE"]])

   ```

## JavaScript npm SCA

The easy way:

```
npm audit
```

If you want to have CVE version you can use `snyk` tool

1. Run

   ```
   npm install -g snyk
   ```

2. Get your github account ready and run

   ```
   snyk auth
   ```

3. Scan

   ```
   snyk test --json-file-output=snyk.json
   ```

4. Run

   ```python
   import csv
   import json
   import csv
   with open('snyk.json", 'r',encoding='utf-8') as file:
           data = json.load(file)

   with  open('snyk.csv', 'w', newline='') as csvfile:
       csv_writer = csv.writer(csvfile)
       csv_writer.writerow(['Component Name', 'Component Version', 'Vulnerability Description', 'CVE ID'])
       for data in data['vulnerabilities']:
           cve = None
           if 'identifiers' in data:
               if 'CVE' in data['identifiers']:
                   if data['identifiers']['CVE']:
                       cve = data['identifiers']['CVE'][0]
           csv_writer.writerow([data["packageName"],data["version"],data["description"],cve])
   ```
