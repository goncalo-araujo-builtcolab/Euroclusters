import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle

@st.cache_data
def load_and_preprocess_data():
    df = pd.read_excel('Iniciativas em Construção Sustentável e Circular (Responses).xlsx').drop(['Timestamp', 'Nome', 'Email'], axis=1)

    # Store original column names
    entity_col = 'Nome da Entidade'
    stakeholder_col = 'Grupo de stakeholders que pertencem à organização (pode ser selecionada mais do que uma)'
    location_col = 'Localização NUT III (pode ser selecionada mais do que uma)'

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

    # Clean and split responses into lists for all relevant columns
    cols_to_split = [location_col, stakeholder_col] + multi_answer_cols
    for col in cols_to_split:
        df[col] = df[col].fillna('').str.split(', ')

    tidy_dfs = []
    
    # Process each question column
    for col in multi_answer_cols:
        # Create temporary dataframe with relevant columns
        temp_df = df[[entity_col, stakeholder_col, col]].copy()
        
        # Explode both stakeholders and the current question's answers
        temp_df = temp_df.explode(stakeholder_col)
        temp_df = temp_df.explode(col)
        
        # Rename columns for consistency
        temp_df = temp_df.rename(columns={
            col: 'Answer',
            stakeholder_col: 'Stakeholder',
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
    # Create a copy of the group column with a temporary name to avoid conflicts
    temp_col = f'temp_{group_col}'
    data = data.copy()
    data[temp_col] = data[group_col]
    
    # Use the temporary column for grouping
    summary = (data.groupby(temp_col, observed=True)
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
    if selected_group == 'Stakeholder':
        group_col = 'Stakeholder'
    elif selected_group == 'Location':
        # Merge with location data for this case
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
    # Create plot
    if selected_chart == 'Sankey':
        # Create node labels and indices
        group_labels = sorted(count_data[group_col].unique())
        answer_labels = answer_order
        
        # Ensure all labels are strings and handle None/NaN values
        group_labels = [str(label) if label is not None else 'Unknown' for label in group_labels]
        answer_labels = [str(label) if label is not None else 'Unknown' for label in answer_labels]
        
        all_nodes = group_labels + answer_labels
        node_indices = {node: idx for idx, node in enumerate(all_nodes)}
        
        # Create links ensuring proper string conversion
        source = []
        target = []
        value = []
        
        for _, row in count_data.iterrows():
            group = str(row[group_col]) if row[group_col] is not None else 'Unknown'
            answer = str(row['Answer']) if row['Answer'] is not None else 'Unknown'
            
            if group in node_indices and answer in node_indices:
                source.append(node_indices[group])
                target.append(node_indices[answer])
                value.append(row['Count'])
        
        # Create color list ensuring we have enough colors
        num_nodes = len(all_nodes)
        colors = palette_mapping[selected_palette]
        # Repeat colors if necessary
        while len(colors) < num_nodes:
            colors = colors + colors
        colors = colors[:num_nodes]
        
        # Create Sankey diagram with improved layout
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=colors
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])
        
        # Update layout with better spacing and size
        fig.update_layout(
            title=dict(
                text=f"{selected_question[:50]}...",
                x=0.5,
                xanchor='center'
            ),
            height=800,
            font=dict(size=12),
            margin=dict(t=100, r=250, b=50, l=250)  # Increased margins for better label visibility
        )
    
    elif selected_chart == 'Pie Chart':
        fig = px.pie(
            count_data,
            names='Answer',
            values='Count',
            color='Answer',
            color_discrete_sequence=palette_mapping[selected_palette],
            title=f"{selected_question[:50]}...",
            category_orders={'Answer': answer_order}
        )
        
        # Update traces with correct percentage display
        fig.update_traces(
            textposition='inside',
            textinfo='percent',
            texttemplate='%{percent}%',  # Remove .1f to show whole percentages
            textfont=dict(size=12)
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.2,
                font=dict(size=10)
            ),
            height=700,
            width=900,
            margin=dict(t=100, b=50, r=200)
        )
    
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
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    
    elif selected_chart == 'Grouped Bar':
        fig = px.bar(
            count_data,
            x=group_col,
            y='Percentage',
            color='Answer',
            text='Percentage',
            barmode='group',
            color_discrete_sequence=palette_mapping[selected_palette],
            title=f"{selected_question[:50]}...",
            category_orders={'Answer': answer_order}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    
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
            title=f"{selected_question[:50]}...",
            category_orders={'Answer': answer_order}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')

    # Common layout updates (except for pie chart and sankey which have their own)
    if selected_chart not in ['Pie Chart', 'Sankey']:
        fig.update_layout(
            height=600,
            xaxis_title=selected_group if selected_chart != 'Horizontal Bar' else 'Percentage (%)',
            yaxis_title='Percentage (%)' if selected_chart != 'Horizontal Bar' else selected_group,
            legend_title='Answers',
            hovermode='closest',
            font=dict(size=12)
        )
    
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Display
    if st.checkbox("Show Raw Data"):
        st.subheader("Processed Data")
        st.dataframe(count_data)

if __name__ == "__main__":
    interactive_analysis()