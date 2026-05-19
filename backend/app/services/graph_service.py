"""
Graph service — Knowledge graph construction, Neo4j sync,
Louvain community detection, and advanced analytics.
"""

import json
import networkx as nx
import numpy as np
from typing import Optional
from pathlib import Path
from app.core.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_DIR
from app.services.llm_service import extract_entities_and_relations
from app.services.embedding_service import get_chunks

# ── Global state ───────────────────────────────────────────────────────────────
_graph: Optional[nx.Graph] = None
_GRAPH_CACHE = DATA_DIR / "knowledge_graph.json"
_partition_cache: Optional[dict] = None


def get_graph() -> Optional[nx.Graph]:
    """Get or load the knowledge graph."""
    global _graph
    if _graph is None:
        _graph = _load_graph_from_cache()
    return _graph


def _load_graph_from_cache() -> Optional[nx.Graph]:
    """Load graph from cached JSON."""
    if _GRAPH_CACHE.exists():
        with open(_GRAPH_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.Graph()
        for node in data.get("nodes", []):
            G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in data.get("edges", []):
            G.add_edge(edge["source"], edge["target"],
                       **{k: v for k, v in edge.items() if k not in ("source", "target")})
        return G
    return None


def _save_graph_to_cache(G: nx.Graph):
    """Save graph to JSON cache."""
    nodes = []
    for n, attrs in G.nodes(data=True):
        node = {"id": n}
        node.update(attrs)
        nodes.append(node)

    edges = []
    for u, v, attrs in G.edges(data=True):
        edge = {"source": u, "target": v}
        edge.update(attrs)
        edges.append(edge)

    with open(_GRAPH_CACHE, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Build Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(use_llm: bool = True) -> dict:
    """Build knowledge graph from ALL chunks using LLM entity/relation extraction."""
    global _graph
    chunks = get_chunks()

    if not chunks:
        return {"status": "error", "message": "No chunks available. Run chunking + embedding first."}

    G = nx.Graph()
    all_entities = []
    all_relations = []
    total = len(chunks)

    print(f"[Graph] Extracting entities from ALL {total} chunks using flan-t5-base...")

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 30:
            continue

        if use_llm:
            result = extract_entities_and_relations(chunk)
        else:
            # Fallback: simple NLP-based extraction
            result = _simple_entity_extraction(chunk)

        for entity in result["entities"]:
            name = entity["name"].strip()
            if name and len(name) > 1:
                if G.has_node(name):
                    G.nodes[name]["count"] = G.nodes[name].get("count", 1) + 1
                else:
                    G.add_node(name, type=entity.get("type", "Concept"),
                               count=1, chunk_index=i)
                all_entities.append(entity)

        for rel in result["relations"]:
            src = rel["source"].strip()
            tgt = rel["target"].strip()
            if src and tgt and src != tgt:
                if G.has_node(src) and G.has_node(tgt):
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] = G[src][tgt].get("weight", 1) + 1
                    else:
                        G.add_edge(src, tgt, type=rel.get("type", "RELATED_TO"), weight=1)
                elif not G.has_node(src):
                    G.add_node(src, type="Concept", count=1)
                    if not G.has_node(tgt):
                        G.add_node(tgt, type="Concept", count=1)
                    G.add_edge(src, tgt, type=rel.get("type", "RELATED_TO"), weight=1)
                all_relations.append(rel)

        if (i + 1) % 10 == 0:
            print(f"[Graph] Processed {i + 1}/{total} chunks, "
                  f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # Remove isolated nodes with low count
    isolated = [n for n in G.nodes() if G.degree(n) == 0 and G.nodes[n].get("count", 1) < 2]
    G.remove_nodes_from(isolated)

    _graph = G
    _save_graph_to_cache(G)

    # Run Louvain for community data
    community_data = detect_communities()

    # Sync to Neo4j with communities
    neo4j_status = "skipped"
    try:
        _sync_to_neo4j(G, community_data.get("partition", {}))
        neo4j_status = "synced"
    except Exception as e:
        neo4j_status = f"failed: {str(e)[:100]}"

    print(f"[Graph] Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return {
        "status": "built",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "neo4j": neo4j_status,
        "chunks_processed": total,
        "communities_detected": community_data.get("cluster_count", 0),
        "modularity": community_data.get("modularity", 0),
    }


def _simple_entity_extraction(text: str) -> dict:
    """Simple regex-based entity extraction fallback."""
    import re
    # Extract capitalized phrases (likely proper nouns / species names)
    pattern = r'\b[A-Z][a-zéèêëàâùûîïôö]+(?:\s+[a-zéèêëàâùûîïôö]+)*\b'
    matches = re.findall(pattern, text)
    entities = []
    seen = set()
    for m in matches:
        m = m.strip()
        if m and m not in seen and len(m) > 2:
            entities.append({"name": m, "type": "Entity"})
            seen.add(m)

    # Create co-occurrence relations
    relations = []
    ents = entities[:8]
    for i in range(len(ents) - 1):
        relations.append({
            "source": ents[i]["name"],
            "target": ents[i + 1]["name"],
            "type": "CO_OCCURS",
        })

    return {"entities": entities[:10], "relations": relations}


# ═══════════════════════════════════════════════════════════════════════════════
#  Neo4j Sync
# ═══════════════════════════════════════════════════════════════════════════════

def _sync_to_neo4j(G: nx.Graph, partition: dict = None):
    """Push the NetworkX graph to Neo4j Aura with community data."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        # Create constraint for fast lookups (ignore if exists)
        try:
            session.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE")
        except Exception:
            pass

        # Create nodes with community info
        for node_id, attrs in G.nodes(data=True):
            community = partition.get(node_id, -1) if partition else -1
            props = {
                "name": node_id,
                "type": attrs.get("type", "Concept"),
                "count": attrs.get("count", 1),
                "community": community,
                "degree": G.degree(node_id),
            }
            session.run(
                "CREATE (n:Concept {name: $name, type: $type, count: $count, "
                "community: $community, degree: $degree})",
                **props
            )

        # Create edges
        for src, tgt, attrs in G.edges(data=True):
            rel_type = attrs.get("type", "RELATED_TO").replace(" ", "_").upper()
            # Sanitize relationship type
            import re
            rel_type = re.sub(r'[^A-Z0-9_]', '_', rel_type)
            if not rel_type:
                rel_type = "RELATED_TO"
            session.run(
                f"MATCH (a:Concept {{name: $src}}), (b:Concept {{name: $tgt}}) "
                f"CREATE (a)-[:{rel_type} {{weight: $weight}}]->(b)",
                src=src, tgt=tgt, weight=attrs.get("weight", 1)
            )

    driver.close()
    print(f"[Neo4j] Synced {G.number_of_nodes()} nodes, {G.number_of_edges()} edges to Neo4j Aura")


def get_neo4j_status() -> dict:
    """Check Neo4j Aura connection and return graph stats from Neo4j."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            node_count = result.single()["node_count"]

            result = session.run("MATCH ()-[r]->() RETURN count(r) AS edge_count")
            edge_count = result.single()["edge_count"]

            # Get community distribution
            result = session.run(
                "MATCH (n) WHERE n.community IS NOT NULL "
                "RETURN n.community AS community, count(n) AS count "
                "ORDER BY count DESC LIMIT 20"
            )
            community_dist = [{"community": r["community"], "count": r["count"]} for r in result]

            # Get sample Cypher query
            result = session.run(
                "MATCH (a)-[r]->(b) RETURN a.name AS source, type(r) AS relation, "
                "b.name AS target LIMIT 10"
            )
            sample_triples = [
                {"source": r["source"], "relation": r["relation"], "target": r["target"]}
                for r in result
            ]

        driver.close()
        return {
            "connected": True,
            "uri": NEO4J_URI,
            "node_count": node_count,
            "edge_count": edge_count,
            "community_distribution": community_dist,
            "sample_triples": sample_triples,
        }
    except Exception as e:
        return {
            "connected": False,
            "uri": NEO4J_URI,
            "error": str(e)[:200],
        }


def execute_cypher(query: str) -> dict:
    """Execute a Cypher query against Neo4j Aura and return results."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            result = session.run(query)
            records = [dict(r) for r in result]

        driver.close()
        return {
            "success": True,
            "query": query,
            "result_count": len(records),
            "results": records[:50],  # Limit output
        }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e)[:300],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph Visualization Data
# ═══════════════════════════════════════════════════════════════════════════════

COMMUNITY_COLORS = [
    "#818cf8", "#34d399", "#fbbf24", "#fb7185", "#22d3ee",
    "#a78bfa", "#f97316", "#60a5fa", "#e879f9", "#4ade80",
    "#f472b6", "#38bdf8", "#facc15", "#c084fc", "#2dd4bf",
]


def get_graph_visualization() -> dict:
    """Return graph data formatted for react-force-graph-2d with community colors."""
    G = get_graph()
    if G is None:
        return {"nodes": [], "edges": [], "legend": {}, "stats": {}}

    # Run Louvain to get community assignments
    partition = _get_partition(G)

    # Calculate node sizes based on degree
    max_degree = max((G.degree(n) for n in G.nodes()), default=1) or 1

    nodes = []
    for n, attrs in G.nodes(data=True):
        community = partition.get(n, 0)
        nodes.append({
            "id": n,
            "label": n,
            "type": attrs.get("type", "Concept"),
            "count": attrs.get("count", 1),
            "degree": G.degree(n),
            "community": community,
            "color": COMMUNITY_COLORS[community % len(COMMUNITY_COLORS)],
            "size": 4 + (G.degree(n) / max_degree) * 22,
        })

    edges = []
    for u, v, attrs in G.edges(data=True):
        # Color edge by source community
        src_comm = partition.get(u, 0)
        tgt_comm = partition.get(v, 0)
        is_inter = src_comm != tgt_comm
        edges.append({
            "source": u,
            "target": v,
            "type": attrs.get("type", "RELATED_TO"),
            "weight": attrs.get("weight", 1),
            "color": "rgba(100,116,139,0.15)" if is_inter else
                     COMMUNITY_COLORS[src_comm % len(COMMUNITY_COLORS)] + "40",
            "interCommunity": is_inter,
        })

    # Node type legend
    types = set(attrs.get("type", "Concept") for _, attrs in G.nodes(data=True))
    legend = {t: f"#{hash(t) % 0xFFFFFF:06x}" for t in types}

    # Community legend
    comm_ids = set(partition.values())
    comm_legend = {}
    for cid in sorted(comm_ids):
        members = [n for n, c in partition.items() if c == cid]
        comm_legend[cid] = {
            "color": COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)],
            "count": len(members),
            "name": f"Communauté {cid + 1}",
        }

    return {
        "nodes": nodes,
        "edges": edges,
        "legend": legend,
        "communityLegend": comm_legend,
        "stats": {
            "nodeCount": G.number_of_nodes(),
            "edgeCount": G.number_of_edges(),
            "communityCount": len(comm_ids),
            "density": round(nx.density(G), 4),
        },
    }


def _get_partition(G: nx.Graph) -> dict:
    """Get Louvain partition for graph (cached)."""
    global _partition_cache
    if _partition_cache is not None:
        return _partition_cache

    if G.number_of_nodes() == 0:
        return {}

    try:
        from community import best_partition
        if not nx.is_connected(G):
            # Run on each component separately
            partition = {}
            offset = 0
            for comp in nx.connected_components(G):
                subG = G.subgraph(comp).copy()
                if subG.number_of_nodes() >= 2:
                    sub_part = best_partition(subG)
                    max_id = max(sub_part.values()) + 1 if sub_part else 0
                    for n, c in sub_part.items():
                        partition[n] = c + offset
                    offset += max_id
                else:
                    for n in subG.nodes():
                        partition[n] = offset
                    offset += 1
        else:
            partition = best_partition(G)
        _partition_cache = partition
        return partition
    except ImportError:
        # Fallback: assign all to community 0
        return {n: 0 for n in G.nodes()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Louvain Community Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_communities() -> dict:
    """Run Louvain community detection on the knowledge graph."""
    try:
        from community import best_partition, modularity as compute_modularity
    except ImportError as e:
        import sys
        try:
            import community
            file_loc = getattr(community, '__file__', 'unknown location')
        except Exception:
            file_loc = 'not found'
            
        raise ImportError(
            f"python-louvain package missing/conflicted.\n"
            f"Server is running using Python: {sys.executable}\n"
            f"Loaded 'community' from: {file_loc}\n"
            "Please ensure you are running Uvicorn from the 'RAGProject' Conda environment, "
            "and run: pip uninstall community python-louvain -y && pip install python-louvain"
        ) from e

    G = get_graph()
    if G is None or G.number_of_nodes() == 0:
        return {"modularity": 0.0, "clusters": [], "cluster_count": 0}

    # Louvain requires connected graph — use largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc).copy()
    else:
        subG = G

    partition = best_partition(subG)
    modularity = compute_modularity(partition, subG)

    # Update partition cache
    global _partition_cache
    # Extend partition to cover all nodes (set isolated component nodes to their own IDs)
    full_partition = dict(partition)
    max_comm = max(partition.values()) + 1 if partition else 0
    for n in G.nodes():
        if n not in full_partition:
            full_partition[n] = max_comm
            max_comm += 1
    _partition_cache = full_partition

    # Group nodes by community
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)

    clusters = []
    for comm_id, members in sorted(communities.items()):
        # Compute intra-cluster density
        sub = subG.subgraph(members)
        intra_edges = sub.number_of_edges()
        possible_edges = len(members) * (len(members) - 1) / 2
        intra_density = round(intra_edges / possible_edges, 4) if possible_edges > 0 else 0

        clusters.append({
            "id": comm_id,
            "name": f"Communauté {comm_id + 1}",
            "members": members,
            "member_count": len(members),
            "color": COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)],
            "key_members": sorted(members, key=lambda n: subG.degree(n), reverse=True)[:5],
            "intra_density": intra_density,
            "intra_edges": intra_edges,
        })

    # Compute inter-community edge counts
    inter_edges = {}
    for u, v in subG.edges():
        cu = partition.get(u, -1)
        cv = partition.get(v, -1)
        if cu != cv:
            key = tuple(sorted([cu, cv]))
            inter_edges[key] = inter_edges.get(key, 0) + 1

    inter_community = [
        {"from": f"C{k[0]+1}", "to": f"C{k[1]+1}", "edges": v}
        for k, v in sorted(inter_edges.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "modularity": round(modularity, 4),
        "clusters": clusters,
        "cluster_count": len(clusters),
        "partition": {n: int(c) for n, c in full_partition.items()},
        "inter_community_edges": inter_community[:20],
        "total_nodes_partitioned": len(partition),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Community-Colored Graph for Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def get_community_graph() -> dict:
    """Return graph with community coloring for the interactive network visualization."""
    G = get_graph()
    if G is None or G.number_of_nodes() == 0:
        return {"nodes": [], "links": [], "communities": [], "stats": {}}

    partition = _get_partition(G)
    max_degree = max((G.degree(n) for n in G.nodes()), default=1) or 1

    # Build community info
    comm_groups = {}
    for n, c in partition.items():
        if c not in comm_groups:
            comm_groups[c] = []
        comm_groups[c].append(n)

    # Nodes
    nodes = []
    for n, attrs in G.nodes(data=True):
        comm = partition.get(n, 0)
        nodes.append({
            "id": n,
            "label": n,
            "type": attrs.get("type", "Concept"),
            "count": attrs.get("count", 1),
            "degree": G.degree(n),
            "community": comm,
            "communityName": f"Communauté {comm + 1}",
            "color": COMMUNITY_COLORS[comm % len(COMMUNITY_COLORS)],
            "val": 2 + (G.degree(n) / max_degree) * 18,
        })

    # Links (react-force-graph uses "links" not "edges")
    links = []
    for u, v, attrs in G.edges(data=True):
        cu = partition.get(u, 0)
        cv = partition.get(v, 0)
        links.append({
            "source": u,
            "target": v,
            "type": attrs.get("type", "RELATED_TO"),
            "weight": attrs.get("weight", 1),
            "interCommunity": cu != cv,
        })

    # Community summaries
    communities = []
    for cid, members in sorted(comm_groups.items()):
        communities.append({
            "id": cid,
            "name": f"Communauté {cid + 1}",
            "count": len(members),
            "color": COMMUNITY_COLORS[cid % len(COMMUNITY_COLORS)],
            "topMembers": sorted(members, key=lambda n: G.degree(n), reverse=True)[:5],
        })

    return {
        "nodes": nodes,
        "links": links,
        "communities": communities,
        "stats": {
            "nodeCount": G.number_of_nodes(),
            "edgeCount": G.number_of_edges(),
            "communityCount": len(comm_groups),
            "density": round(nx.density(G), 4),
            "avgDegree": round(sum(G.degree(n) for n in G.nodes()) / G.number_of_nodes(), 2),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Advanced Graph Analytics
# ═══════════════════════════════════════════════════════════════════════════════

def get_graph_analytics() -> dict:
    """Return advanced graph metrics."""
    G = get_graph()
    if G is None or G.number_of_nodes() == 0:
        return {"centrality": {}, "density": 0.0, "semantic_paths": [], "recommendations": []}

    # Centrality metrics
    degree_cent = nx.degree_centrality(G)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]

    betweenness = {}
    if G.number_of_nodes() > 2 and nx.is_connected(G):
        betweenness = nx.betweenness_centrality(G)
    elif G.number_of_nodes() > 2:
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        betweenness = nx.betweenness_centrality(subG)

    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

    # Closeness centrality
    closeness = {}
    if G.number_of_nodes() > 2:
        if nx.is_connected(G):
            closeness = nx.closeness_centrality(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subG)
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]

    # PageRank
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        top_pagerank = []

    # Density
    density = nx.density(G)

    # Connected components
    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)

    # Degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    degree_dist = {}
    for d in degrees:
        degree_dist[d] = degree_dist.get(d, 0) + 1
    degree_histogram = [
        {"degree": d, "count": c}
        for d, c in sorted(degree_dist.items())
    ]

    # Find semantic paths (shortest paths between top nodes)
    semantic_paths = []
    top_nodes = [n for n, _ in top_degree[:5]]
    for i in range(len(top_nodes)):
        for j in range(i + 1, len(top_nodes)):
            try:
                path = nx.shortest_path(G, top_nodes[i], top_nodes[j])
                semantic_paths.append({
                    "from": top_nodes[i],
                    "to": top_nodes[j],
                    "path": path,
                    "length": len(path) - 1,
                })
            except nx.NetworkXNoPath:
                pass

    # Recommendations
    recommendations = []
    # Find bridge nodes
    if G.number_of_nodes() > 3:
        for n, bc in top_betweenness[:3]:
            recommendations.append(f"Bridge node: '{n}' (betweenness: {bc:.3f})")
    # Find dense clusters
    communities = detect_communities()
    for cluster in communities.get("clusters", [])[:3]:
        recommendations.append(
            f"Cluster '{cluster['name']}' has {cluster['member_count']} members "
            f"(density: {cluster.get('intra_density', 'N/A')})"
        )

    return {
        "centrality": {
            "degree": [{"node": n, "score": round(s, 4)} for n, s in top_degree],
            "betweenness": [{"node": n, "score": round(s, 4)} for n, s in top_betweenness],
            "closeness": [{"node": n, "score": round(s, 4)} for n, s in top_closeness],
            "pagerank": [{"node": n, "score": round(s, 6)} for n, s in top_pagerank],
        },
        "density": round(density, 4),
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "components": len(components),
        "component_sizes": component_sizes[:10],
        "degree_distribution": degree_histogram[:20],
        "semantic_paths": semantic_paths[:5],
        "recommendations": recommendations,
    }
