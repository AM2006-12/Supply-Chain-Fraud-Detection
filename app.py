"""
Supply Chain Fraud Detection - Streamlit Web App
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from graph_builder import GraphBuilder
from fraud_detector import FraudDetector
from visualizer import FraudVisualizer

# Page config
st.set_page_config(
    page_title="Supply Chain Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
# Defaults for user options
if 'use_patterns' not in st.session_state:
    st.session_state.use_patterns = True
if 'use_statistical' not in st.session_state:
    st.session_state.use_statistical = True
if 'use_gnn' not in st.session_state:
    st.session_state.use_gnn = False
if 'fast_analysis' not in st.session_state:
    st.session_state.fast_analysis = True


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Supply Chain Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Graph Neural Network System")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Detection Settings")
        use_patterns = st.checkbox("Graph Pattern Analysis", value=st.session_state.use_patterns)
        use_statistical = st.checkbox("Statistical Anomaly Detection", value=st.session_state.use_statistical)
        use_gnn = st.checkbox("GNN Model (Demo)", value=st.session_state.use_gnn, 
                             help="GNN model requires training data")
        fast_analysis = st.checkbox("Fast Analysis (recommended)", value=st.session_state.fast_analysis, help="Use faster heuristics to avoid long-running pattern searches")
        
        # Persist choices to session state so other functions can access them
        st.session_state.use_patterns = use_patterns
        st.session_state.use_statistical = use_statistical
        st.session_state.use_gnn = use_gnn
        st.session_state.fast_analysis = fast_analysis
        
        st.subheader("Visualization Settings")
        max_nodes = st.slider("Max Nodes to Display", 20, 200, 100)
        min_risk = st.slider("Minimum Risk Score", 0.0, 1.0, 0.0, 0.1)
        
        st.markdown("---")
        st.markdown("**About**")
        st.info("""
        This system uses Graph Neural Networks and pattern analysis 
        to detect fraud in supply chain transactions.
        
        **Features:**
        - Circular trading detection
        - Shell company identification
        - Anomaly detection
        - Interactive visualizations
        """)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Data", "📊 Analysis", "🌐 Network View", "📋 Reports"])
    
    with tab1:
        upload_data_tab()
    
    with tab2:
        if st.session_state.data_loaded:
            analysis_tab(use_patterns, use_statistical, use_gnn)
        else:
            st.warning("⚠️ Please upload transaction data first")
    
    with tab3:
        # Allow Network View to trigger analysis automatically if data is loaded
        network_view_tab(max_nodes, min_risk)
    
    with tab4:
        if st.session_state.analysis_done:
            reports_tab()
        else:
            st.warning("⚠️ Please run analysis first")

def _run_analysis_from_session():
    """Run analysis using data and options stored in st.session_state."""
    try:
        processor = st.session_state.get('processor')
        df = st.session_state.get('df')
        if df is None:
            st.warning("No data available to analyze.")
            return None

        graph_builder = GraphBuilder()
        G = graph_builder.build_graph(df)

        detector = FraudDetector(graph_builder)
        results = detector.detect_fraud(
            use_gnn=st.session_state.get('use_gnn', False),
            use_statistical=st.session_state.get('use_statistical', True),
            use_graph_patterns=st.session_state.get('use_patterns', True),
            fast_mode=st.session_state.get('fast_analysis', True)
        )

        st.session_state.graph_builder = graph_builder
        st.session_state.detector = detector
        st.session_state.results = results
        st.session_state.report = detector.generate_report(results)
        st.session_state.analysis_done = True
        return results
    except Exception as e:
        st.error(f"Error running analysis: {e}")
        st.session_state.analysis_done = False
        return None


def upload_data_tab():
    st.header("📤 Upload Transaction Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Required CSV Format:**
        - `from_company`: Sender company name
        - `to_company`: Receiver company name
        - `amount`: Transaction amount
        - `date`: Transaction date (YYYY-MM-DD)
        - `category`: (Optional) Transaction category
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your transaction data"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                processor = DataProcessor()
                df = processor.load_data(uploaded_file)
                
                # Clean data
                df = processor.clean_data(df)
                df = processor.compute_basic_features(df)
                
                # Store in session state
                st.session_state.df = df
                st.session_state.processor = processor
                st.session_state.data_loaded = True
                
                st.success(f"✅ Successfully loaded {len(df)} transactions")
                
                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show statistics
                stats = processor.get_summary_stats(df)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Total Transactions", f"{stats['total_transactions']:,}")
                with col_b:
                    st.metric("Unique Companies", stats['unique_companies'])
                with col_c:
                    st.metric("Total Volume", f"₹{stats['total_volume']/10000000:.1f}Cr")

                # Run analysis automatically after loading
                with st.spinner("Running analysis... (auto)"):
                    _run_analysis_from_session()
                if st.session_state.analysis_done:
                    st.success("✅ Analysis completed automatically")
                else:
                    st.warning("⚠️ Analysis did not complete automatically; you can run it from the Analysis tab")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")

    with col2:
        st.markdown("**Quick Actions**")
        if st.button("📥 Generate Sample Data"):
            try:
                from data.generate_data import TransactionGenerator
                generator = TransactionGenerator(num_companies=50, num_transactions=1000)
                df = generator.generate_dataset()

                # Use existing processor to clean and compute features
                processor = DataProcessor()
                df = processor.clean_data(df)
                df = processor.compute_basic_features(df)

                # Store in session state
                st.session_state.df = df
                st.session_state.processor = processor
                st.session_state.data_loaded = True

                st.success(f"✅ Generated {len(df)} sample transactions")

                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Note: Analysis is not run automatically for generated sample data
                st.info("Run analysis from the Analysis tab when you're ready to analyze the generated dataset.")
            except Exception as e:
                st.error(f"❌ Error generating sample data: {str(e)}")

def analysis_tab(use_patterns, use_statistical, use_gnn):
    st.header("📊 Fraud Detection Analysis")
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing transactions..."):
            try:
                # Build graph
                graph_builder = GraphBuilder()
                G = graph_builder.build_graph(st.session_state.df)
                
                # Run fraud detection
                detector = FraudDetector(graph_builder)
                results = detector.detect_fraud(
                    use_gnn=use_gnn,
                    use_statistical=use_statistical,
                    use_graph_patterns=use_patterns,
                    fast_mode=st.session_state.get('fast_analysis', True)
                )
                
                # Store results
                st.session_state.graph_builder = graph_builder
                st.session_state.detector = detector
                st.session_state.results = results
                # Also generate and store a summarized report for the UI
                st.session_state.report = detector.generate_report(results)
                st.session_state.analysis_done = True
                
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error("❌ An error occurred during analysis!")
                st.exception(e)
                st.session_state.analysis_done = False # Do not proceed if analysis failed

        st.rerun()
    
    if st.session_state.analysis_done:
        results = st.session_state.results
        # Use stored report if available; otherwise generate on-the-fly
        report = st.session_state.get('report') or st.session_state.detector.generate_report(results)
        
        # Summary metrics
        st.subheader("Detection Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Entities", report['summary']['total_entities'])
        with col2:
            st.metric("Transactions", report['summary']['total_transactions'])
        with col3:
            st.metric("High Risk", report['summary']['high_risk_count'], 
                     delta_color="inverse")
        with col4:
            st.metric("Suspicious Cycles", report['summary']['suspicious_cycles'],
                     delta_color="inverse")
        with col5:
            st.metric("Fraud Rings", report['summary']['fraud_rings'],
                     delta_color="inverse")
        
        # High-risk entities
        st.subheader("🚨 Top High-Risk Entities")
        if results['high_risk_entities']:
            risk_df = pd.DataFrame(results['high_risk_entities'])
            st.dataframe(
                risk_df[['rank', 'node', 'score']].head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No high-risk entities detected")
        
        # Visualizations
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Risk Score Distribution")
            visualizer = FraudVisualizer(st.session_state.graph_builder)
            fig = visualizer.plot_risk_distribution(results['final_scores'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col_b:
            st.subheader("Top Risky Entities")
            fig = visualizer.plot_top_risks(results['final_scores'], top_n=10)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cycles
        if results['cycles']:
            st.subheader("🔄 Suspicious Circular Trading Patterns")
            for i, cycle in enumerate(results['cycles'][:3]):
                with st.expander(f"Cycle {i+1} - Risk Score: {cycle['risk_score']:.2f}"):
                    st.write(f"**Length:** {cycle['length']} companies")
                    st.write(f"**Total Amount:** ₹{cycle['total_amount']:,.0f}")
                    st.write(f"**Companies involved:** {' → '.join(cycle['nodes'])}")
                    
                    # Visualize cycle
                    fig = visualizer.visualize_cycle(cycle['nodes'])
                    st.plotly_chart(fig, use_container_width=True)

def network_view_tab(max_nodes, min_risk):
    st.header("🌐 Interactive Network Visualization")

    # If analysis hasn't been run yet but data is loaded, run it now
    if not st.session_state.get('analysis_done', False):
        if st.session_state.get('data_loaded', False):
            with st.spinner("Running analysis to prepare network..."):
                _run_analysis_from_session()
            if not st.session_state.get('analysis_done'):
                st.error("Analysis failed or timed out. Please run analysis manually from the Analysis tab.")
                return
        else:
            st.warning("⚠️ Please upload data and run analysis first")
            return

    results = st.session_state.results
    graph_builder = st.session_state.graph_builder
    visualizer = FraudVisualizer(graph_builder)
    
    viz_type = st.radio("Visualization Type", ["PyVis (Interactive)", "Plotly (Static)"])
    
    if viz_type == "PyVis (Interactive)":
        st.info("💡 Tip: Drag nodes, zoom, and click on nodes/edges for details")
        
        net = visualizer.create_interactive_network(
            results['final_scores'],
            max_nodes=max_nodes,
            min_risk_score=min_risk
        )
        
        # Save and display
        net.save_graph('temp_network.html')
        with open('temp_network.html', 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=800)
        
    else:
        fig = visualizer.create_plotly_network(
            results['final_scores'],
            max_nodes=max_nodes
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity search
    st.subheader("🔍 Entity Details")
    
    entities = list(graph_builder.G.nodes())
    selected_entity = st.selectbox("Select an entity to view details", entities)
    
    if selected_entity:
        details = st.session_state.detector.get_entity_details(
            selected_entity,
            results
        )
        
        if details:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{details['risk_score']:.2f}")
            with col2:
                st.metric("Total Received", f"₹{details['total_received']/1000000:.1f}M")
            with col3:
                st.metric("Total Sent", f"₹{details['total_sent']/1000000:.1f}M")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write("**Incoming Transactions:**")
                if details['incoming_transactions']:
                    st.dataframe(pd.DataFrame(details['incoming_transactions']))
                else:
                    st.write("None")
            
            with col_b:
                st.write("**Outgoing Transactions:**")
                if details['outgoing_transactions']:
                    st.dataframe(pd.DataFrame(details['outgoing_transactions']))
                else:
                    st.write("None")
            
            if details['participates_in_cycles'] > 0:
                st.warning(f"⚠️ This entity participates in {details['participates_in_cycles']} suspicious cycles!")

def reports_tab():
    st.header("📋 Audit Reports")
    
    results = st.session_state.results
    detector = st.session_state.detector
    
    report = detector.generate_report(results)
    
    # Download report
    report_text = generate_text_report(report)
    
    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name="fraud_detection_report.txt",
        mime="text/plain"
    )
    
    # Display report
    st.subheader("Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entities Analyzed", report['summary']['total_entities'])
    with col2:
        st.metric("High-Risk Entities", report['summary']['high_risk_count'])
    with col3:
        fraud_pct = (report['summary']['high_risk_count'] / report['summary']['total_entities']) * 100
        st.metric("Risk Percentage", f"{fraud_pct:.1f}%")
    
    # Top risks
    st.subheader("Top 10 High-Risk Entities")
    if report['top_risks']:
        risk_table = pd.DataFrame(report['top_risks'])[['rank', 'node', 'score']]
        risk_table.columns = ['Rank', 'Entity', 'Risk Score']
        st.table(risk_table)
    
    # Suspicious patterns
    st.subheader("Suspicious Patterns Detected")
    
    if report['top_cycles']:
        st.write("**Circular Trading Patterns:**")
        for i, cycle in enumerate(report['top_cycles']):
            st.write(f"{i+1}. {len(cycle['nodes'])} companies, ₹{cycle['total_amount']:,.0f} total")
    
    if report['suspicious_communities']:
        st.write("**Fraud Rings:**")
        for i, comm in enumerate(report['suspicious_communities']):
            st.write(f"{i+1}. {comm['size']} entities, ₹{comm['total_volume']:,.0f} volume")

def generate_text_report(report):
    """Generate downloadable text report"""
    text = "SUPPLY CHAIN FRAUD DETECTION REPORT\n"
    text += "="*50 + "\n\n"
    
    text += "EXECUTIVE SUMMARY\n"
    text += "-"*50 + "\n"
    text += f"Total Entities Analyzed: {report['summary']['total_entities']}\n"
    text += f"Total Transactions: {report['summary']['total_transactions']}\n"
    text += f"High-Risk Entities: {report['summary']['high_risk_count']}\n"
    text += f"Suspicious Cycles: {report['summary']['suspicious_cycles']}\n"
    text += f"Fraud Rings: {report['summary']['fraud_rings']}\n\n"
    
    text += "TOP 10 HIGH-RISK ENTITIES\n"
    text += "-"*50 + "\n"
    for risk in report['top_risks']:
        text += f"{risk['rank']}. {risk['node']}: {risk['score']:.3f}\n"
    
    return text

if __name__ == "__main__":
    main()
