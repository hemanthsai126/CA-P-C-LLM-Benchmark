# Official California P&C / producer licensing URLs

Curated for this benchmark project. **These are outlines, bulletins, and scheduling—not scraped MCQ banks.** CDI and PSI occasionally change paths; if a link breaks, start from [Exam info](https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/).

## California Department of Insurance (CDI)

| Resource | URL |
|----------|-----|
| Agents & brokers (industry hub) | https://www.insurance.ca.gov/0200-industry/ |
| Insurance license exam info (PSI scheduling, remote testing overview) | https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/ |
| Exam time limits and number of questions (includes P&C 150 questions / 3 hours) | https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/ExamTimesandQuestion.cfm |
| Remote proctored exam FAQ | https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/Remote-Testing-Frequently-Asked-Question.cfm |
| Applying for a license (resident / non-resident / entity) | https://www.insurance.ca.gov/0200-industry/0020-apply-license/ |
| Individual resident license (forms, steps; exam outline links often live here) | https://www.insurance.ca.gov/0200-industry/0020-apply-license/0100-indiv-resident/index.cfm |
| License application after PSI (Sircon link; lists Property & Casualty Broker-Agent exam) | https://www.insurance.ca.gov/0200-industry/0020-apply-license/referral-from-psi.cfm |
| Prelicensing / continuing education | https://www.insurance.ca.gov/0200-industry/0030-seek-pre-lic/ |
| Producer licensing FAQ | https://www.insurance.ca.gov/0200-industry/0090-faq/ |

## PSI (California insurance exam vendor)

| Resource | URL |
|----------|-----|
| PSI California (CADI) test-taker scheduling portal | https://test-takers.psiexams.com/cadi |
| PSI marketing/info page for California CADI | https://www.psiexams.com/test-takers/cadi/ |
| Candidate Information Bulletin (PDF via PSI; educational objectives—linked from CDI exam times page) | http://candidate.psiexams.com/bulletin/display_bulletin.jsp?ro=yes&actionname=83&bulletinid=506&bulletinurl=.pdf |

## Application (Sircon / Vertafore)

| Resource | URL |
|----------|-----|
| Individual license application (per CDI / PSI flow) | https://www.sircon.com/products/apply.jsp |

## Notes

- **Property and Casualty Broker-Agent** is called out explicitly on CDI’s PSI referral page as a combined examination type.
- **Exam content outlines** are usually PDFs linked from the individual resident / exam sections; if CDI updates filenames (e.g. dated bulletins), use the exam info page and the PSI bulletin link above to find the current file.
- **PSI bulletin URL:** automated `GET` may return an HTML shell (login / redirect) instead of the PDF. Open the same URL in a browser and use **Save as** for the real bulletin if needed.
- **Snapshots on disk:** run `python notebooks/fetch_official_snapshots.py` from the repo root; files land in `notebooks/extracted/fetched_cache_official/`.
- Do not assume **practice exam websites** are legal to scrape; this list is **official channels only**.
