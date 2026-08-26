"""Self-contained D3-force graph visualization, rendered as a Streamlit
component (an iframe, via `st.components.v1.html`).

No custom bidirectional Streamlit component (that needs a JS build
toolchain) — the iframe just redraws fresh from whatever nodes/edges
Python hands it on each poll tick. Simulation state doesn't persist
across reruns; node positions are re-seeded deterministically from the
node id (so a node lands in roughly the same spot every redraw instead
of jumping randomly) and a short elastic "pop" on entrance is what reads
as motion, not a continuously running simulation.
ponytail: a real bidirectional component (streamlit.components.v1.declare_component)
would let the simulation persist across polls instead of re-seeding — worth it
only if the re-seed jitter becomes visually distracting.

v2 note: user confirmed this tradeoff is fine for now (re-seeded-per-poll,
not continuous physics) — revisit only if the redraw jitter actually bothers
someone watching it live, not preemptively.
"""

import json

import streamlit as st

_INK_NAVY = "#0B1330"
_INK_SOFT = "#9AA1C4"
_SOURCE_COLOR = "#5B6494"
_TOOLTIP_BG = "#1C2454"


def render_graph(
    nodes: list[dict],
    edges: list[dict],
    active_dim: str | None,
    dim_colors: dict[str, str],
    height: int = 480,
) -> None:
    payload = json.dumps(
        {
            "nodes": nodes,
            "edges": edges,
            "activeDim": active_dim,
            "dimColors": dim_colors,
            "sourceColor": _SOURCE_COLOR,
        }
    )
    html = f"""
    <div id="graph-root" style="width:100%;height:{height}px;background:{_INK_NAVY};
         border-radius:20px;position:relative;overflow:hidden;
         box-shadow:0 12px 40px rgba(0,0,0,.35);">
      <div id="tooltip" style="position:absolute;opacity:0;visibility:hidden;max-width:320px;
           background:{_TOOLTIP_BG};color:{_INK_SOFT};font:12px 'IBM Plex Sans',system-ui,sans-serif;
           padding:8px 10px;border-radius:10px;pointer-events:none;z-index:10;
           box-shadow:0 4px 16px rgba(0,0,0,.4);
           transition:opacity 0.15s ease;"></div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
    <script>
    const data = {payload};
    const root = document.getElementById('graph-root');
    const width = root.clientWidth, height = {height};

    const svg = d3.select('#graph-root').append('svg')
        .attr('width', width).attr('height', height);

    function seed(id, mult) {{
      let h = 0;
      for (let i = 0; i < id.length; i++) h = (h * mult + id.charCodeAt(i)) >>> 0;
      return (h % 1000) / 1000;
    }}

    // Each redraw is a fresh iframe with no memory of the last one (see
    // module docstring), so "don't move nodes that were already visible"
    // can't be done by tracking what's new — there's nothing to compare
    // against. Anchoring every node to its id-seeded position instead gets
    // the same visible effect: a given node's anchor is identical on every
    // redraw, so it settles back to roughly the same spot regardless of how
    // many new neighbors just joined, rather than the whole layout
    // re-balancing around them.
    const nodeById = new Map();
    data.nodes.forEach(n => {{
      n.seedX = seed(n.id, 31) * width;
      n.seedY = seed(n.id, 17) * height;
      n.x = n.seedX;
      n.y = n.seedY;
      n.r = n.kind === 'source_sentence' ? 3.5 : (5 + n.level * 2.5);
      n.color = n.dimension ? (data.dimColors[n.dimension] || '#888') : data.sourceColor;
      nodeById.set(n.id, n);
    }});
    const links = data.edges
      .filter(e => nodeById.has(e.child) && nodeById.has(e.parent))
      .map(e => ({{...e, source: e.child, target: e.parent}}));

    const sim = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(38).strength(0.1))
      .force('charge', d3.forceManyBody().strength(-20))
      .force('collision', d3.forceCollide(d => d.r + 3))
      .force('anchorX', d3.forceX(d => d.seedX).strength(0.35))
      .force('anchorY', d3.forceY(d => d.seedY).strength(0.35))
      .alpha(0.5)
      .alphaDecay(0.06);

    function opacityFor(dim) {{
      if (!data.activeDim) return dim ? 0.85 : 0.9;
      if (!dim) return 0.35;
      return dim === data.activeDim ? 1 : 0.12;
    }}

    const link = svg.append('g').selectAll('line').data(links).join('line')
      .attr('stroke', d => data.dimColors[d.dimension] || '#888')
      .attr('stroke-width', d => 0.6 + d.weight * 2)
      .attr('stroke-opacity', d => opacityFor(d.dimension))
      .style('transition', 'stroke-opacity 0.2s ease');

    const node = svg.append('g').selectAll('circle').data(data.nodes).join('circle')
      .attr('r', 0)
      .attr('fill', d => d.color)
      .style('transition', 'opacity 0.2s ease')
      .attr('opacity', d => opacityFor(d.dimension))
      .style('cursor', d => d.kind === 'source_sentence' ? 'pointer' : 'default')
      .on('mouseenter', (event, d) => {{
        if (!d.text) return;
        const tip = document.getElementById('tooltip');
        tip.textContent = d.text;
        tip.style.left = Math.min(event.offsetX + 12, width - 330) + 'px';
        tip.style.top = (event.offsetY + 12) + 'px';
        tip.style.visibility = 'visible';
        tip.style.opacity = '1';
      }})
      .on('mouseleave', () => {{
        const tip = document.getElementById('tooltip');
        tip.style.opacity = '0';
        tip.style.visibility = 'hidden';
      }});

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) {{
      node.attr('r', d => d.r);
    }} else {{
      node.transition().duration(700).ease(d3.easeElasticOut.amplitude(1).period(0.5))
        .attr('r', d => d.r);
    }}

    sim.on('tick', () => {{
      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      node.attr('cx', d => d.x).attr('cy', d => d.y);
    }});
    </script>
    """
    st.components.v1.html(html, height=height + 4, scrolling=False)
