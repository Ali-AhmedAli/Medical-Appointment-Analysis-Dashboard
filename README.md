# 🏥 Medical Appointment No-Show Analysis Dashboard

An interactive data dashboard built with Dash and Plotly to analyze medical appointment attendance patterns from a Brazilian healthcare dataset.

## 📊 Dataset Overview

This dashboard analyzes over **110,000 medical appointment records** from Brazil, including:

- **Patient Demographics**: Age, Gender, Neighborhood
- **Appointment Details**: Scheduled vs. Appointment dates, Day of week
- **Medical Conditions**: Diabetes, Hypertension, Alcoholism, Handicap
- **Communication**: SMS received status
- **Socioeconomic**: Scholarship status (low-income support)
- **Outcome**: Whether patients showed up for appointments

## 🎯 Dashboard Features

### 📈 Interactive Visualizations

1. **No-Show vs Show-Up Distribution** - Pie chart showing attendance rates
2. **Age Distribution Analysis** - Histogram with attendance breakdown by age
3. **Day of Week Analysis** - Stacked bar chart showing appointment patterns
4. **Gender Analysis** - Dual charts comparing appointments and no-show rates by gender
5. **Medical Conditions Impact** - Grouped bar chart showing how conditions affect attendance
6. **Neighborhood Analysis** - Neighborhoods with no-show rates
7. **Days Until Appointment** - Distribution of scheduling lead times
8. **SMS Impact Analysis** - Comparison of SMS effectiveness

### 🎛️ Interactive Filters

- **Age Group**: Filter by specific age ranges (0-18, 19-35, 36-50, 51-65, 65+)
- **Gender**: Filter by Male/Female
- **Neighborhood**: Filter by specific neighborhoods
- **Medical Conditions**: Filter by patients with specific conditions

### 📊 Key Metrics Display

- **Total Appointments**: Real-time count based on filters
- **No-Show Rate**: Percentage of missed appointments
- **Average Age**: Mean age of patients
- **SMS Received Rate**: Percentage of patients who received SMS reminders

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files in your directory:
   # - app.py
   # - KaggleV2-May-2016.csv
   # - requirements.txt
   # - README.md
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   python app.py
   ```

4. **Access the dashboard**
   - Open your web browser
   - Navigate to: `http://localhost:8050`
   - The dashboard will load automatically

## 📋 Requirements

The dashboard uses the following Python libraries:

- **Dash** (3.1.1) - Web framework for building analytical web applications
- **Plotly** (6.2.0) - Interactive plotting library
- **Pandas** (2.3.1) - Data manipulation and analysis
- **NumPy** (1.26.4) - Numerical computing
- **dash-bootstrap-components** (2.0.3) Adds Bootstrap styling and layout to Dash applications
 
## 🎨 Dashboard Design

### Color Scheme
- **Green** (#27ae60) - Show-up appointments
- **Red** (#e74c3c) - No-show appointments
- **Blue** (#3498db) - Total appointments
- **Orange** (#f39c12) - SMS metrics

### Layout Features
- **Responsive Design**: Adapts to different screen sizes
- **Professional Styling**: Clean, modern interface
- **Interactive Elements**: Hover effects and dynamic updates
- **Clear Navigation**: Intuitive filter controls

## 📊 Key Insights

The dashboard reveals important patterns in medical appointment attendance:

1. **No-Show Patterns**: Overall no-show rate and factors affecting it
2. **Age Impact**: How patient age influences attendance
3. **Gender Differences**: Attendance patterns between males and females
4. **Day of Week Effects**: Which days have higher/lower attendance
5. **Medical Conditions**: How chronic conditions affect attendance
6. **Geographic Patterns**: Neighborhood-based attendance variations
7. **Communication Impact**: Effectiveness of SMS reminders
8. **Scheduling Patterns**: Relationship between lead time and attendance

## 🔧 Technical Implementation

### Data Processing
- **Date Conversion**: Scheduled and appointment dates converted to datetime
- **Feature Engineering**: Created age groups, day of week, lead time calculations
- **Binary Encoding**: Converted categorical variables to binary for analysis

### Interactive Features
- **Real-time Filtering**: All charts update based on user selections
- **Dynamic Metrics**: Key statistics update with filter changes
- **Responsive Charts**: Charts adapt to filtered data automatically

### Performance Optimizations
- **Efficient Data Loading**: Single data load with multiple filtered views
- **Optimized Callbacks**: Minimal redundant calculations
- **Memory Management**: Efficient data handling for large dataset

## 🎓 Educational Value

This dashboard demonstrates:

- **Data Visualization Best Practices**: Clear, informative charts
- **Interactive Dashboard Design**: User-friendly interface
- **Real-world Data Analysis**: Practical application of data science
- **Python Web Development**: Building web applications with Dash
- **Statistical Analysis**: Understanding patterns in healthcare data

## 📝 Code Organization

- **🔧 Main Application**: Initializes the Dash app and layout
- **📊 Data Processing**: Loads and filters appointment data for analysis  
- **🎯 Callback Functions**: Updates charts based on user selections  
- **📈 Visualization Components**: Builds the individual charts, including:
  - No-Show by Medical Condition  
  - No-Show by Neighborhood  
  - Attendance by Gender & Age  
  - Interactive Geographical Map
## 📄 License

This project is created for educational purposes and data analysis demonstration.

---

**Built using Dash and Plotly** 