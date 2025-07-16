# Meta Video Classification Interface

A Streamlit-based interface for reviewing and annotating video content classifications with advanced scene detection and caption quality assessment.

## 🎯 Features

- 🎬 **Video Content Review**: Frame-by-frame analysis of Meta video content
- 📊 **Classification Audit**: Verify and correct AI classification decisions (CSBS vs Safe)
- 📝 **Caption Quality Assessment**: Comprehensive caption evaluation with accuracy, completeness, and relevance metrics
- 🔍 **Scene Detection Review**: Assess video scene detection quality and frame extraction
- 📈 **Real-time Analytics**: Track annotation progress and session statistics
- 💾 **Data Export**: Export detailed annotations to CSV for further analysis
- 🌐 **Multi-language Support**: Automatic translation for non-English content
- 🔄 **Duplicate Detection**: Automatic removal of duplicate content for clean datasets


### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/meta-video-classification-interface.git
cd meta-video-classification-interface
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up Google Cloud authentication:**
```bash
# Method 1: Application Default Credentials (Recommended)
gcloud auth application-default login

# Method 2: Service Account Key
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Access the interface:**
   - Open your browser to `http://localhost:8501`

## 📋 Usage Guide

### 1. **Load Data**
- Select a date from the date picker
- Choose sample size (Quick Select: 1-500, or Custom: up to 10,000)
- Select classification type (0=CSBS, 100=Safe, Any=Mixed)
- Click "Load Data"

### 2. **Navigate Content**
- Use **Previous/Next** buttons to navigate between records
- Use **Jump to record** to go to a specific item
- Track progress with "Record X of Y" counter

### 3. **Review Video Frames**
- Navigate through video frames using **Previous Frame/Next Frame**
- View frame timestamps and metadata
- Assess frame quality and relevance

### 4. **Annotation Workflow**

#### **Caption Audit:**
- **Accuracy**: Rate how well captions describe the content
- **Completeness**: Assess if captions cover all important elements
- **Relevance**: Evaluate caption relevance to video content
- **Issues**: Select specific problems (wrong descriptions, missing elements, etc.)
- **Quality Score**: Rate overall caption quality (1-10 scale)
- **Improvements**: Suggest better captions when needed

#### **Classification Review:**
- **Data Collection Quality**: Rate data pipeline quality
- **Classification Accuracy**: Verify if AI classification is correct
- **Scene Detection Quality**: Assess frame extraction and scene detection

### 5. **Export Results**
- Click **"Add Correction"** to save annotations
- Use **"Save Corrections"** to download CSV file
- Track session progress in sidebar statistics

