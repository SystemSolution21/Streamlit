1. Text Elements:
   - `st.title()` -> `<h1>` tag
   - `st.header()` -> `<h2>` tag
   - `st.subheader()` -> `<h3>` tag
   - `st.markdown()` -> Converts markdown to HTML elements
   - `st.text()` -> `<p>` tag with monospace font
   - `st.code()` -> `<code>` tag
   - `st.caption()` -> Small text with gray color

2. Input Widgets:
   - `st.text_input()` -> `<input type="text">`
   - `st.number_input()` -> `<input type="number">`
   - `st.text_area()` -> `<textarea>`
   - `st.date_input()` -> Custom date picker
   - `st.time_input()` -> Custom time picker
   - `st.file_uploader()` -> `<input type="file">`
   - `st.color_picker()` -> `<input type="color">`

3. Button Elements:
   - `st.button()` -> `<button>`
   - `st.download_button()` -> `<a>` with download attribute
   - `st.checkbox()` -> `<input type="checkbox">`
   - `st.radio()` -> `<input type="radio">`
   - `st.selectbox()` -> `<select>`
   - `st.multiselect()` -> `<select multiple>`

4. Layout Elements:
   - `st.sidebar` -> Side navigation panel
   - `st.columns()` -> CSS Grid/Flexbox layout
   - `st.expander()` -> Collapsible section
   - `st.container()` -> `<div>` container
   - `st.tabs()` -> Tab navigation interface

5. Media Elements:
   - `st.image()` -> `<img>`
   - `st.video()` -> `<video>`
   - `st.audio()` -> `<audio>`

6. Data Display:
   - `st.dataframe()` -> Interactive data table
   - `st.table()` -> Static HTML table
   - `st.metric()` -> Custom metric display
   - `st.json()` -> Formatted JSON viewer
   - `st.plotly_chart()` -> Interactive Plotly chart

7. Status Elements:
   - `st.error()` -> Red error message box
   - `st.warning()` -> Yellow warning message box
   - `st.info()` -> Blue info message box
   - `st.success()` -> Green success message box
   - `st.exception()` -> Exception traceback display
   - `st.progress()` -> `<progress>` bar

8. Custom CSS Classes:
   ```css
   .stApp -> Main app container
   .stMarkdown -> Markdown content
   .stButton -> Button elements
   .stTextInput -> Text input fields
   .stSelectbox -> Dropdown selects
   .stDataFrame -> DataFrame display
   .stTable -> Table display
   .stMetric -> Metric display
   .stAlert -> Alert messages
   .stSidebar -> Sidebar container
   .stTabs -> Tab container
   .stTab -> Individual tab
   .stExpander -> Expandable section
   ```

9. Common CSS Customization:
   ```css
   /* Example of custom styling */
   .stApp {
       background-color: #f0f2f6;
   }
   
   .stButton button {
       background-color: #4CAF50;
       color: white;
       border-radius: 5px;
   }
   
   .stTextInput input {
       border: 2px solid #4CAF50;
       border-radius: 5px;
   }
   
   .stMetric {
       background-color: white;
       padding: 15px;
       border-radius: 10px;
       box-shadow: 0 2px 4px rgba(0,0,0,0.1);
   }
   ```