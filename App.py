# Page Configuration
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle
from plotly.subplots import make_subplots

# Set page config to wide layout
st.set_page_config(layout="wide")

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the survey data."""
    # Read the Excel file and drop unnecessary columns
    df = pd.read_excel('Iniciativas em Construção Sustentável e Circular (Responses).xlsx')

    # Store original column names
    entity_col = 'Nome da Entidade'
    stakeholder_col = 'Grupo de stakeholders que pertencem à organização (pode ser selecionada mais do que uma)'
    location_col = 'Localização NUT III (pode ser selecionada mais do que uma)'

    # Define multi-answer columns
    multi_answer_cols = [
        'Se tivesse 60 000€ para o desenvolvimento de um novo serviço ou produto relacionado com a reutilização de materiais (excluindo investimentos), como usaria este valor? (Escolha até 3 opções)',
        'Que barreiras existentes identifica na implementação e no desenvolvimento de um novo produto ou serviço para a reutilização de materiais?  (Escolha até 3 opções)',
        'Que estratégias de circularidade / reutilização já estão implementadas na sua entidade?  (É possível selecionar mais do que uma opção).',
        'Que ações internacionais considera necessárias para o desenvolvimento de práticas mais circulares / reutilização? (É possível selecionar mais do que uma opção).',
        'De forma a desenvolver práticas de circularidade / reutilização na sua entidade, em quais das seguintes opções estaria interessado? (É possível selecionar mais do que uma opção).',
        'No âmbito do B4PIC, quais das seguintes atividades considera ser mais importante para a sua organização? (É possível selecionar mais do que uma opção).',
        'Em que tipo de projetos gostaria de participar? (É possível selecionar mais do que uma opção).',
        'Que indicadores de sustentabilidade poderiam apoiar a sua atividade? (É possível selecionar mais do que uma opção).',
    ]

    def clean_and_split_answers(text, is_multiple=False):
        """
        Clean and split answers while preserving commas within parentheses.
        Parameters:
            text (str): The text to clean and split
            is_multiple (bool): Whether this is a multiple-choice field (like stakeholders/locations)
        Returns:
            list: List of cleaned answers
        """
        if pd.isna(text) or text == '':
            return []
        
        # Split by the primary delimiter first ('., ')
        answers = text.split('., ')
        
        # Clean each answer
        cleaned_answers = []
        for answer in answers:
            # Remove trailing periods and commas
            answer = answer.strip(' .,')
            if answer:
                if is_multiple:
                    # For multiple-choice fields, split by comma and clean each one
                    items = [s.strip() for s in answer.split(', ')]
                    cleaned_answers.extend(items)
                else:
                    cleaned_answers.append(answer)
        
        # Remove duplicates while preserving order for multiple-choice fields
        if is_multiple:
            seen = set()
            cleaned_answers = [x for x in cleaned_answers if not (x in seen or seen.add(x))]
        
        return cleaned_answers

    # Clean and split responses for all relevant columns
    cols_to_split = [location_col, stakeholder_col] + multi_answer_cols
    for col in cols_to_split:
        # Pass is_multiple flag for both stakeholder and location columns
        is_multiple = col in [stakeholder_col, location_col]
        df[col] = df[col].apply(lambda x: clean_and_split_answers(x, is_multiple=is_multiple))

    # Process each question into a tidy format
    tidy_dfs = []
    for col in multi_answer_cols:
        # Create temporary dataframe with relevant columns
        temp_df = df[[entity_col, stakeholder_col, col]].copy()
        
        # Explode both stakeholders and answers
        temp_df = temp_df.explode(stakeholder_col)
        temp_df = temp_df.explode(col)
        
        # Rename columns for consistency
        temp_df = temp_df.rename(columns={
            col: 'Answer',
            stakeholder_col: 'Stakeholder',
            entity_col: 'Entity'
        })
        
        # Add question column
        temp_df['Question'] = col
        
        tidy_dfs.append(temp_df)

    # Combine all processed dataframes
    tidy_df = pd.concat(tidy_dfs, ignore_index=True)

    # Clean whitespace and drop empty rows
    tidy_df['Answer'] = tidy_df['Answer'].str.strip()
    tidy_df = tidy_df.dropna(subset=['Answer'])
    tidy_df = tidy_df[tidy_df['Answer'] != '']
    
    # Handle location data separately
    location_df = df[[entity_col, stakeholder_col, location_col]].copy()
    location_df = location_df.explode(stakeholder_col)
    location_df = location_df.explode(location_col)
    location_df = location_df.rename(columns={
        stakeholder_col: 'Stakeholder',
        location_col: 'Location',
        entity_col: 'Entity'
    })
    
    # Clean location data
    location_df['Location'] = location_df['Location'].fillna('')
    location_df = location_df[location_df['Location'] != '']
    
    return tidy_df, location_df

def display_group_table(data, group_col):
    """Display summary table for the selected grouping"""
    # Create a temporary column name that won't conflict with existing columns
    temp_col = f'temp_{group_col}_grouping'
    
    # Make a copy of the data to avoid modifying the original
    temp_data = data.copy()
    temp_data[temp_col] = temp_data[group_col]
    
    # Use the temporary column for grouping
    summary = (temp_data.groupby(temp_col, observed=True)
               .agg({
                   'Answer': 'count',
                   'Entity': 'nunique'
               })
               .reset_index()
               .rename(columns={
                   temp_col: group_col,
                   'Answer': 'Total Responses',
                   'Entity': 'Unique Entities'
               }))
    
    return summary

def truncate_text(text, max_length=40):
    """Helper function to truncate text for legend entries"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'

def update_legend_text(fig, max_length=40):
    """Update legend text for all traces in a figure"""
    for trace in fig.data:
        if hasattr(trace, 'name'):
            trace.name = truncate_text(trace.name, max_length)

def create_chart_layout(chart_type, count_data, group_col):
    """Create consistent layout settings for charts"""
    # Calculate dynamic height based on data
    num_categories = len(count_data[group_col].unique())
    base_height = 500  # minimum height
    height_per_category = 40  # height per category
    
    if chart_type == 'Horizontal Bar':
        chart_height = max(base_height, num_categories * height_per_category)
    else:
        chart_height = base_height
    
    return dict(
        height=chart_height,
        margin=dict(r=150, t=50, b=50, l=50),  # Reduced right margin for legend
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=8),  # Smaller font size for legend
            tracegroupgap=5  # Reduce gap between legend items
        ),
        font=dict(size=12)  # Main font size
    )

def interactive_analysis():
    st.title("Interactive Circular Economy Initiatives Analysis")
    
    # Load data
    tidy_df, location_df = load_and_preprocess_data()
    
    # Sidebar Controls
    st.sidebar.header("Chart Configuration")
    
    questions = sorted(tidy_df['Question'].unique())
    selected_question = st.sidebar.selectbox("Select Question", questions)
    
    group_options = ['Location', 'Entity', 'Stakeholder', 'Total']
    selected_group = st.sidebar.selectbox("Group By", group_options)
    
    chart_types = ['Stacked Bar', 'Treemap', 'Horizontal Bar', 'Sankey', 'Interactive Pie']  # Added Faceted Pie option
    selected_chart = st.sidebar.selectbox("Chart Type", chart_types)
    
    
    palettes = ['Viridis', 'Plasma', 'Portland']
    selected_palette = st.sidebar.selectbox("Color Palette", palettes)
    
    # Data Processing
    question_df = tidy_df[tidy_df['Question'] == selected_question].copy()
    
    # Clean answer text
    question_df['Answer'] = question_df['Answer'].apply(
        lambda x: x.split('(exemplo:')[0].strip() if isinstance(x, str) and 'exemplo:' in x else x
    )
    
    # Handle grouping
    if selected_group == 'Stakeholder':
        group_col = 'Stakeholder'
    elif selected_group == 'Location':
        question_df = pd.merge(
            question_df,
            location_df[['Entity', 'Location']].drop_duplicates(),
            on='Entity',
            how='left'
        )
        group_col = 'Location'
    elif selected_group == 'Entity':
        group_col = 'Entity'
    else:  # Total
        question_df['Total'] = 'Total'
        group_col = 'Total'

    # Calculate metrics and sort by values
    count_data = question_df.groupby([group_col, 'Answer']).size().reset_index(name='Count')
    
    # Sort answers by total count
    answer_totals = count_data.groupby('Answer')['Count'].sum().sort_values(ascending=False)
    answer_order = list(answer_totals.index)
    
    # Calculate percentages for each group
    count_data['Percentage'] = count_data.groupby(group_col)['Count'].transform(
        lambda x: (x / x.sum()) * 100
    )
    
    # Display group summary
    st.header("Group Summary")
    if selected_group != 'Total':
        summary_table = display_group_table(question_df, group_col)
        st.dataframe(summary_table)
    else:
        total_responses = len(question_df)
        unique_entities = question_df['Entity'].nunique()
        col1, col2 = st.columns(2)
        col1.metric("Total Responses", total_responses)
        col2.metric("Unique Entities", unique_entities)

    # Visualization
    palette_mapping = {
        'Viridis': px.colors.sequential.Viridis,
        'Plasma': px.colors.sequential.Plasma,
        'Portland': px.colors.sequential.Plotly3
    }
    
    st.header("Interactive Visualization")
    
    # Create plot
    if selected_chart == 'Sankey':
        # Create node labels and indices
        group_labels = sorted(count_data[group_col].unique())
        answer_labels = answer_order
        
        all_nodes = list(group_labels) + list(answer_labels)
        node_indices = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create links
        source = [node_indices[row[group_col]] for _, row in count_data.iterrows()]
        target = [node_indices[row['Answer']] for _, row in count_data.iterrows()]  # Removed the + len(group_labels)
        value = count_data['Count'].tolist()
        
        # Generate enough colors for all nodes
        base_colors = palette_mapping[selected_palette]
        num_colors_needed = len(all_nodes)
        
        # Create a color sequence by repeating and interpolating if necessary
        if len(base_colors) < num_colors_needed:
            repeat_times = (num_colors_needed // len(base_colors)) + 1
            extended_colors = base_colors * repeat_times
            node_colors = extended_colors[:num_colors_needed]
        else:
            node_colors = base_colors[:num_colors_needed]
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="lightgray", width=0.5),
                label=[truncate_text(node) for node in all_nodes],
                color=node_colors,  # This should now apply to all nodes
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=[f"rgba({int(int(node_colors[s][1:3], 16))}, {int(int(node_colors[s][3:5], 16))}, {int(int(node_colors[s][5:7], 16))}, 0.4)" for s in source]
            )
        )])

    elif selected_chart == 'Interactive Pie':
        # Add dropdown for specific group selection
        groups = sorted(count_data[group_col].unique())
        selected_group_value = st.selectbox(
            f"Select {selected_group}",
            groups,
            key='pie_group_selector'
        )
        
        # Filter data for selected group
        group_data = count_data[count_data[group_col] == selected_group_value]
        
        # Create color mapping for consistency
        answer_color_map = {
            answer: color
            for answer, color in zip(
                answer_order,
                palette_mapping[selected_palette][:len(answer_order)]
            )
        }
        
        # Ensure consistent order and colors
        ordered_data = pd.DataFrame({'Answer': answer_order}).merge(
            group_data,
            on='Answer',
            how='left'
        ).fillna(0)
        
        # Get colors in the same order as the data
        colors = [answer_color_map[ans] for ans in ordered_data['Answer']]
        
        # Create single pie chart
        fig = go.Figure(data=[go.Pie(
            labels=ordered_data['Answer'],
            values=ordered_data['Count'],
            marker_colors=colors,
            textposition='inside',
            textinfo='percent',
            #texttemplate='%{percent*100:.1f}%',
            hovertemplate="%{label}<br>%{value} responses<br>%{percent*100:.1f}%<extra></extra>"
        )])
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text=f"{selected_question[:50]}...<br><sub>{selected_group}: {selected_group_value}</sub>",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=8)
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_chart == 'Stacked Bar':
        fig = px.bar(
            count_data,
            x=group_col,
            y='Percentage',
            color='Answer',
            text='Percentage',
            barmode='stack',
            color_discrete_sequence=palette_mapping[selected_palette],
            title=f"{selected_question[:50]}...",
            category_orders={'Answer': answer_order}
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='inside',
            width=0.8
        )
        update_legend_text(fig)
        fig.update_layout(**create_chart_layout(selected_chart, count_data, group_col))
    
    
    elif selected_chart == 'Treemap':
        fig = px.treemap(
            count_data,
            path=[group_col, 'Answer'],
            values='Count',
            color='Answer',
            color_discrete_sequence=palette_mapping[selected_palette],
            title=f"{selected_question[:50]}..."
        )
        
        layout = create_chart_layout(selected_chart, count_data, group_col)
        layout.update(height=700)  # Specific height for treemap
        fig.update_layout(**layout)
    
    elif selected_chart == 'Horizontal Bar':
        fig = px.bar(
            count_data,
            y=group_col,
            x='Percentage',
            color='Answer',
            orientation='h',
            text='Percentage',
            color_discrete_sequence=palette_mapping[selected_palette],
            title=f"{selected_question[:50]}...",
            category_orders={'Answer': answer_order}
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%', 
            textposition='inside',
            width=0.8
        )
        update_legend_text(fig)
        fig.update_layout(**create_chart_layout(selected_chart, count_data, group_col))
    
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Display
    if st.checkbox("Show Raw Data"):
        st.subheader("Processed Data")
        st.dataframe(count_data)

if __name__ == "__main__":
    interactive_analysis()