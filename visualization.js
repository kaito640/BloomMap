let svg, width, height, radius, innerRadius, middleRadius, outerRadius, color;
let clusterData;

function initializeVisualization() {
  svg = d3.select("#visualization");
  width = +svg.attr("width");
  height = +svg.attr("height");
  radius = Math.min(width, height) / 2 - 20;
  
  innerRadius = +document.getElementById("innerRadiusStart").value / 100 * radius;
  middleRadius = +document.getElementById("innerRadiusEnd").value / 100 * radius;
  outerRadius = +document.getElementById("outerRadiusStart").value / 100 * radius;

  color = d3.scaleOrdinal(d3.schemeCategory10);

  svg.selectAll("*").remove(); // Clear previous visualization
}

function createDonutRings(data) {
  const g = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  // Add white background for central area
  g.append("circle")
    .attr("r", innerRadius)
    .attr("fill", "white");

  const pie = d3.pie()
    .value(1)
    .sort(null);

  const arc = d3.arc();

  // Inner ring (darker colors, cluster labels)
  const innerArc = arc.innerRadius(innerRadius).outerRadius(middleRadius);
  g.selectAll("path.inner")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "inner")
    .attr("d", innerArc)
    .attr("fill", (d, i) => color(i))
    .attr("stroke", "white")
    .attr("stroke-width", 1);

  // Middle ring (pale, transparent background)
  const middleArc = arc.innerRadius(middleRadius).outerRadius(outerRadius);
  g.selectAll("path.middle")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "middle")
    .attr("d", middleArc)
    .attr("fill", (d, i) => d3.color(color(i)).brighter(1.5))
    .attr("stroke", "none")
    .attr("opacity", 0.3);

  // Outer ring (thin, clean look)
  const outerArc = arc.innerRadius(outerRadius).outerRadius(outerRadius + 2);
  g.selectAll("path.outer")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "outer")
    .attr("d", outerArc)
    .attr("fill", (d, i) => d3.color(color(i)).darker(0.5))
    .attr("stroke", "none");

  // Add cluster labels to inner ring
  g.selectAll("text.cluster-label")
    .data(pie(data))
    .enter()
    .append("text")
    .attr("class", "cluster-label")
    .attr("transform", d => {
      const [x, y] = arc.innerRadius((innerRadius + middleRadius) / 2).centroid(d);
      return `translate(${x},${y}) rotate(${(d.startAngle + d.endAngle) / 2 * 180 / Math.PI - 90})`;
    })
    .attr("text-anchor", "middle")
    .attr("fill", "white")
    .attr("font-size", "40px")
    .attr("font-weight", "bold")
    .text(d => d.data);

  return {pie, middleArc};
}

function createVoronoiTreemap(data, clipPolygon, clusterIndex) {
  const voronoiTreemap = d3.voronoiTreemap()
    .clip(clipPolygon)
    .minWeightRatio(0.01)
    .prng(Math.random);

  const rootNode = d3.hierarchy({children: data})
    .sum(d => d.importance);

  voronoiTreemap(rootNode);

  return rootNode;
}

function drawVoronoiTreemap(treemap, x, y, clusterIndex, clusterName) {
  const g = svg.append("g")
    .attr("transform", `translate(${x},${y})`);

  g.selectAll("path")
    .data(treemap.descendants().filter(d => d.depth > 0))
    .enter()
    .append("path")
    .attr("d", d => `M${d.polygon.join("L")}Z`)
    .attr("fill", clusterIndex === 'global' ? "#f0f0f0" : d3.color(color(clusterIndex)).brighter(1.5))
    .attr("stroke", clusterIndex === 'global' ? "#333" : color(clusterIndex))
    .attr("stroke-width", 1);

  g.selectAll("text")
    .data(treemap.descendants().filter(d => d.depth > 0))
    .enter()
    .append("text")
    .attr("x", d => d.polygon.reduce((acc, point) => acc + point[0], 0) / d.polygon.length)
    .attr("y", d => d.polygon.reduce((acc, point) => acc + point[1], 0) / d.polygon.length)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("font-size", d => Math.min(50, d.data.importance * 60) + "px")
    .attr("fill", clusterIndex === 'global' ? "#333" : color(clusterIndex))
    .text(d => d.data.theme);

  // Add cluster label
  if (clusterIndex !== 'global') {
    g.append("text")
      .attr("class", "cluster-label")
      .attr("x", 0)
      .attr("y", 0)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "central")
      .attr("font-size", "40px")
      .attr("font-weight", "bold")
      .attr("fill", color(clusterIndex))
      .text(clusterName);
  }

  // Add hover-over tooltip
  const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

  g.selectAll("path")
    .on("mouseover", function(event, d) {
      tooltip.transition()
        .duration(200)
        .style("opacity", .9);
      tooltip.html(`${d.data.theme}<br/>Importance: ${d.data.importance.toFixed(2)}`)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
    })
    .on("mouseout", function(d) {
      tooltip.transition()
        .duration(500)
        .style("opacity", 0);
    });
}

function getClipPolygon(startAngle, endAngle, innerRadius, outerRadius) {
  const step = Math.PI / 180;
  const points = [];

  for (let angle = startAngle; angle <= endAngle; angle += step) {
    points.push([Math.cos(angle) * innerRadius, Math.sin(angle) * innerRadius]);
  }
  for (let angle = endAngle; angle >= startAngle; angle -= step) {
    points.push([Math.cos(angle) * outerRadius, Math.sin(angle) * outerRadius]);
  }

  return points;
}

async function createVisualization() {
  initializeVisualization();

  const topNWords = +document.getElementById("topNWords").value;

  const clusters = Array.from(clusterData.keys()).filter(k => k !== 'global');
  const {pie, middleArc} = createDonutRings(clusters);

  // Create Voronoi treemaps for each cluster
  pie(clusters).forEach((d, i) => {
    const cluster = clusterData.get(d.data);
    if (cluster) {
      const clipPolygon = getClipPolygon(d.startAngle, d.endAngle, middleRadius, outerRadius);
      const treemap = createVoronoiTreemap(cluster.slice(0, topNWords), clipPolygon, d.index);
      const centroid = middleArc.centroid(d);
      drawVoronoiTreemap(treemap, width/2 + centroid[0], height/2 + centroid[1], d.index, d.data);
    }
  });

  // Create Voronoi treemap for global cluster in the center
  const globalCluster = clusterData.get('global');
  if (globalCluster) {
    const clipPolygon = getClipPolygon(0, 2 * Math.PI, 0, innerRadius);
    const treemap = createVoronoiTreemap(globalCluster.slice(0, topNWords), clipPolygon, 'global');
    drawVoronoiTreemap(treemap, width/2, height/2, 'global', 'Global');
  }
}

d3.csv("processed_themes.csv").then(data => {
  clusterData = d3.group(data, d => d.cluster);

  console.log("Processed cluster data:", clusterData);

  createVisualization();

  document.getElementById("updateViz").addEventListener("click", () => {
    createVisualization().catch(error => {
      console.error("Error updating visualization:", error);
    });
  });

  document.getElementById("downloadSVG").addEventListener("click", () => {
    const svgData = new XMLSerializer().serializeToString(svg.node());
    const svgBlob = new Blob([svgData], {type: "image/svg+xml;charset=utf-8"});
    const svgUrl = URL.createObjectURL(svgBlob);
    const downloadLink = document.createElement("a");
    downloadLink.href = svgUrl;
    downloadLink.download = "radial_clusters.svg";
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  });
}).catch(error => {
  console.error("Error loading data or drawing visualization:", error);
  alert("Error loading data. Please check the console for more information.");
});
