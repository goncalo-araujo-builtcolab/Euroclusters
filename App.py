import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_excel('Iniciativas em Construção Sustentável e Circular (Responses).xlsx').drop(['Timestamp', 'Nome', 'Email'], axis=1)

    # Store location column name for reference
    location_col = 'Localização NUT III (pode ser selecionada mais do que uma)'
    entity_col = 'Nome da Entidade'
    stakeholder_col = 'Grupo de stakeholders que pertencem à organização (pode ser selecionada mais do que uma)'
    
    # Define question columns (excluding location)
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

    # Split all multi-answer columns
    all_cols = multi_answer_cols + [location_col, stakeholder_col]
    for col in all_cols:
        df[col] = df[col].fillna('').str.split(', ')

    tidy_dfs = []
    
    # Process each question column
    for col in multi_answer_cols:
        # Create temporary dataframe
        temp_df = df[[entity_col, stakeholder_col, location_col, col]].copy()
        
        # Explode all multiple-choice columns
        temp_df = temp_df.explode(stakeholder_col)
        temp_df = temp_df.explode(location_col)
        temp_df = temp_df.explode(col)
        
        # Rename columns for consistency
        temp_df = temp_df.rename(columns={
            col: 'Answer',
            stakeholder_col: 'Stakeholder',
            location_col: 'Location',
            entity_col: 'Entity'
        })
        
        temp_df['Question'] = col
        tidy_dfs.append(temp_df)

    # Combine all DataFrames
    tidy_df = pd.concat(tidy_dfs, ignore_index=True)

    # Clean whitespace and drop empty rows
    tidy_df['Answer'] = tidy_df['Answer'].str.strip()
    tidy_df = tidy_df.dropna(subset=['Answer'])
    tidy_df = tidy_df[tidy_df['Answer'] != '']
    
    return tidy_df

def display_group_table(data, group_col):
    """Display summary table for the selected grouping"""
    # First create summary without reset_index
    summary = data.groupby(group_col).agg({
        'Answer': 'count',
        'Entity': 'nunique'
    })
    
    # Rename columns before reset_index
    summary.columns = ['Total Responses', 'Unique Entities']
    
    # Now reset index safely
    summary = summary.reset_index()
    return summary

def interactive_analysis():
    st.title("Interactive Circular Economy Initiatives Analysis")
    
    # Load data
    tidy_df = load_and_preprocess_data()
    
    # Sidebar Controls
    st.sidebar.header("Chart Configuration")
    
    questions = sorted(tidy_df['Question'].unique())
    selected_question = st.sidebar.selectbox("Select Question", questions)
    
    group_options = ['Location', 'Entity', 'Stakeholders', 'Total']
    selected_group = st.sidebar.selectbox("Group By", group_options)
    
    chart_types = ['Stacked Bar', 'Grouped Bar', 'Pie Chart', 'Treemap', 'Horizontal Bar', 'Sankey']
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
    if selected_group == 'Stakeholders':
        group_col = 'Stakeholder'
        merged_df = question_df
    elif selected_group == 'Location':
        group_col = 'Location'
        merged_df = question_df
    elif selected_group == 'Entity':
        group_col = 'Entity'
        merged_df = question_df
    else:  # Total
        merged_df = question_df.copy()
        merged_df['Total'] = 'Total'
        group_col = 'Total'

    # Calculate metrics
    count_data = merged_df.groupby([group_col, 'Answer']).size().reset_index(name='Count')
    count_data['Percentage'] = count_data.groupby(group_col)['Count'].apply(
        lambda x: x / x.sum() * 100
    ).reset_index(drop=True)
    
    # Display group summary
    st.header("Group Summary")
    if selected_group != 'Total':
        summary_table = display_group_table(merged_df, group_col)
        st.dataframe(summary_table)
    else:
        total_responses = len(merged_df)
        unique_entities = merged_df['Entity'].nunique()
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
    if selected_chart == 'Pie Chart':
        fig = px.pie(
            count_data,
            names='Answer',
            values='Count',
            color='Answer',
            color_discrete_sequence=palette_mapping[selected_palette],
            title=f"{selected_question[:50]}..."
        )
        # Remove the automatic percentage text
        fig.update_traces(textposition='inside', textinfo='percent+label')
    elif selected_chart == 'Sankey':
        # Prepare data for Sankey diagram
        source = []
        target = []
        value = []
        
        # Create node labels
        unique_groups = merged_df[group_col].unique()
        unique_answers = merged_df['Answer'].unique()
        
        # Create mapping for node indices
        group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
        answer_to_idx = {answer: idx + len(unique_groups) for idx, answer in enumerate(unique_answers)}
        
        # Create links data
        for _, row in count_data.iterrows():
            source.append(group_to_idx[row[group_col]])
            target.append(answer_to_idx[row['Answer']])
            value.append(row['Count'])
        
        # Create node labels
        node_labels = list(unique_groups) + list(unique_answers)
        
        # Get colors from the selected palette
        node_colors = px.colors.sample_colorscale(palette_mapping[selected_palette], 
                                                 len(node_labels))
        
        # Create Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = node_labels,
                color = node_colors,
                hoverlabel = dict(bgcolor = 'white'),
                hovertemplate = '%{label}<br>' +
                              'Total: %{value}<extra></extra>'
            ),
            link = dict(
                source = source,
                target = target,
                value = value,
                hoverlabel = dict(bgcolor = 'white'),
                hovertemplate = 'From: %{source.label}<br>' +
                              'To: %{target.label}<br>' +
                              'Count: %{value}<extra></extra>'
            ),
            arrangement = 'snap'
        )])
    else:
        if selected_chart == 'Stacked Bar':
            fig = px.bar(
                count_data,
                x=group_col,
                y='Percentage',
                color='Answer',
                text='Percentage',
                barmode='stack',
                color_discrete_sequence=palette_mapping[selected_palette],
                title=f"{selected_question[:50]}..."
            )
        elif selected_chart == 'Grouped Bar':
            fig = px.bar(
                count_data,
                x=group_col,
                y='Percentage',
                color='Answer',
                text='Percentage',
                barmode='group',
                color_discrete_sequence=palette_mapping[selected_palette],
                title=f"{selected_question[:50]}..."
            )
        elif selected_chart == 'Treemap':
            fig = px.treemap(
                count_data,
                path=[group_col, 'Answer'],
                values='Count',
                color='Answer',
                color_discrete_sequence=palette_mapping[selected_palette],
                title=f"{selected_question[:50]}..."
            )
        elif selected_chart == 'Horizontal Bar':
            fig = px.bar(
                count_data,
                y=group_col,
                x='Percentage',
                color='Answer',
                orientation='h',
                text='Percentage',
                color_discrete_sequence=palette_mapping[selected_palette],
                title=f"{selected_question[:50]}..."
            )

    # Update text format for regular charts (excluding Treemap and Sankey)
    if selected_chart not in ['Treemap', 'Sankey']:
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')

    # Common layout updates
    fig.update_layout(
        height=600,
        xaxis_title=selected_group if selected_chart != 'Horizontal Bar' else 'Percentage (%)',
        yaxis_title='Percentage (%)' if selected_chart != 'Horizontal Bar' else selected_group,
        legend_title='Answers',
        hovermode='closest',
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Display
    if st.checkbox("Show Raw Data"):
        st.subheader("Processed Data")
        st.dataframe(count_data)

if __name__ == "__main__":
    interactive_analysis()