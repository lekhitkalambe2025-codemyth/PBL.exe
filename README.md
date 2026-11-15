#  WhatsApp Image Steganography Detector

### Detecting Hidden Information in WhatsApp Images Using Image Analysis and Machine Learning

##  About the Project

The **WhatsApp Image Steganography Detector** is a cybersecurity-focused project designed to identify potentially hidden information within images shared through WhatsApp.

Steganography is a technique used to conceal secret messages or information inside digital files, particularly images. Unlike encryption, which makes information unreadable, steganography attempts to hide the very existence of the information.

This project combines **statistical image analysis and Machine Learning** to examine image characteristics, identify suspicious patterns, and classify images as clean or potentially containing hidden data.

##  Project Objectives

- Detect potential hidden information embedded in digital images.
- Analyze image properties using statistical and pixel-level techniques.
- Identify suspicious Least Significant Bit (LSB) patterns.
- Apply Machine Learning for image classification.
- Generate detection results with confidence scores.
- Improve awareness of image-based steganography and digital security.

##  Key Features

- **Image Analysis:** Examines pixel values, image characteristics and statistical patterns.
- **LSB Analysis:** Identifies unusual patterns in Least Significant Bit distributions.
- **Noise Analysis:** Examines irregularities in image noise.
- **Histogram Analysis:** Evaluates pixel intensity distributions.
- **Feature Extraction:** Extracts 24 numerical features from images.
- **Machine Learning Detection:** Uses a Random Forest classifier.
- **Image Classification:** Distinguishes between clean images and potentially steganographic images.
- **Confidence Score:** Provides a confidence-based prediction.
- **Browser Extension:** Supports the project’s browser-based detection workflow.

##  Technologies Used

| Technology | Purpose |
|---|---|
| HTML5 | Web page structure |
| CSS3 | User interface and styling |
| JavaScript | Frontend functionality |
| Python | Backend development |
| Flask | Web application framework |
| OpenCV | Image processing |
| Pillow | Image handling |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning |
| Random Forest | Image classification |

##  Methodology

The project follows a systematic image detection pipeline:

1. **Image Input:** The user provides an image for analysis.
2. **Preprocessing:** The image is loaded and converted into a suitable format.
3. **Feature Extraction:** Relevant image characteristics are extracted, including LSB patterns, entropy, noise and histogram features.
4. **Statistical Analysis:** Extracted characteristics are examined for suspicious patterns.
5. **Machine Learning Classification:** The trained Random Forest model evaluates the extracted features.
6. **Result Generation:** The application displays the predicted classification and confidence score.

##  Machine Learning Model

The project uses a **Random Forest Classifier** for image classification.

- Algorithm: Random Forest
- Extracted Features: 24
- Classification: Clean Image / Steganographic Image
- Approach: Statistical Feature Extraction + Supervised Machine Learning

##  Model Performance

The following results were obtained during testing on the project's demonstration dataset.

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 95.83% |
| ROC-AUC Score | 0.9931 |
| Dataset Size | 240 images |

**Note:** These results are based on a synthetic demonstration dataset. Performance on real-world WhatsApp-transferred images may differ and requires further validation.

##  Applications

- Cybersecurity and digital forensics
- Image-based steganography detection
- Digital communication security
- Suspicious image investigation
- Academic research and experimentation

##  Future Scope

- Support for additional image formats.
- Improved detection of advanced steganography techniques.
- Testing on larger and more diverse real-world datasets.
- Integration of deep learning models.
- Improved browser extension capabilities.
- Enhanced detection reports and visualization.

##  Academic Project Details

| Detail | Information |
|---|---|
| Project Title | WhatsApp Image Steganography Detector |
| Project Type | Project-Based Learning (PBL) |
| Group Number | 55 |
| Semester | III |
| Project Guide | Prof. Firdous Sadaf |
| Department | Computer Science and Engineering |
| Institution | Symbiosis Institute of Technology, Nagpur |

##  Contributors

This project was developed as part of the Semester III Project-Based Learning (PBL) curriculum.

**Group 55**

##  Disclaimer

This project is developed for educational and research purposes. Detection results indicate potential steganographic activity and should not be considered definitive proof of hidden information.

---

**Keywords:** Steganography, WhatsApp Security, Cybersecurity, Image Processing, LSB Analysis, Machine Learning, Random Forest, Digital Forensics, Python, Flask.
