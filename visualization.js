let svg, width, height, radius, innerRadius, middleRadius, outerRadius, color;
let clusterData = {}, volumeData = {};

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
  URL.revokeObjectURL(svgUrl);
});


function initializeVisualization() {
  svg = d3.select("#visualization");
  width = +svg.attr("width");
  height = +svg.attr("height");
  radius = Math.min(width, height) / 2 - 80;
  
  innerRadius = +document.getElementById("innerRadiusStart").value / 100 * radius;
  middleRadius = +document.getElementById("innerRadiusEnd").value / 100 * radius;
  outerRadius = +document.getElementById("outerRadiusStart").value / 100 * radius;

  color = d3.scaleOrdinal(d3.schemeCategory10);

  svg.selectAll("*").remove(); // Clear previous visualization
}

function createDonutRings(data, vizGroup, volumeData) {
  console.log("Data passed to createDonutRings:", data);
  console.log("Volume data:", volumeData);

  const g = vizGroup.append("g");

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
    .attr("opacity", 0.2);

  // Define volumeScale
  const volumes = Object.values(volumeData).filter(v => typeof v === 'number' && !isNaN(v));
  console.log("Filtered volumes:", volumes);
  
  if (volumes.length === 0) {
    console.error("No valid volume data found");
    return {pie, middleArc};
  }


  // Filter out the "global" value
  const localVolumes = Object.values(volumeData).filter((value, index) => {
    // Skip the first value (global)
    return index !== 0;
  });

  const volumeScale = d3.scaleLinear()
    .domain([0, d3.max(localVolumes)*1.2])
    .range([0, outerRadius - middleRadius]);

  // Duplicate middle ring for volume representation
  g.selectAll("path.volume")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "volume")
    .attr("d", (d) => {
      const volume = volumeData[d.data] || 0;
      console.log(`Volume for ${d.data}:`, volume);
      const outerRadiusAdjusted = middleRadius + volumeScale(volume);
      return arc.innerRadius(middleRadius).outerRadius(outerRadiusAdjusted)(d);
    })
    .attr("fill", (d, i) => d3.color(color(i)).brighter(1.5))
    .attr("stroke", "none")
    .attr("opacity", 0.8);

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

  // Add cluster labels and volume count near inner donut
  g.selectAll("text.cluster-label")
    .data(pie(data))
    .enter()
    .append("text")
    .attr("class", "cluster-label")
    .attr("transform", d => {
      const [x, y] = arc.innerRadius(innerRadius + 5).centroid(d);
      return `translate(${x},${y})`;
    })
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("fill", "white")
    .attr("font-size", "12px")
    .attr("font-weight", "bold")
    .text((d) => {
      const volume = volumeData[d.data] || 0;
      return `${d.data}\n${volume}`;
    });

  return {pie, middleArc};
}

function createVoronoiTreemap(data, clipPolygon, clusterIndex) {
  const voronoiTreemap = d3.voronoiTreemap()
    .clip(clipPolygon)
    .minWeightRatio(0.01)
    .prng(Math.random);

  const rootNode = d3.hierarchy({children: data})
    .sum(d => d[1]);  // Use the importance value directly

  voronoiTreemap(rootNode);

  return rootNode;
}

function drawVoronoiTreemap(treemap, x, y, clusterIndex, clusterName, vizGroup) {
  const g = vizGroup.append("g")
    .attr("transform", `translate(${x},${y})`);

  g.selectAll("path")
    .data(treemap.descendants().filter(d => d.depth > 0))
    .enter()
    .append("path")
    .attr("d", d => `M${d.polygon.join("L")}Z`)
    //.attr("fill", clusterIndex === 'global' ? "#f0f0f0" : color(clusterIndex)) //d3.color(color(clusterIndex)).brighter(1.5) 
    .attr("fill", clusterIndex === 'global' ? "#333333" : color(clusterIndex)) // Dark grey for global
    //.attr("stroke", clusterIndex === 'global' ? "#d0d0d0" : "white")
    .attr("stroke", "white")
    //.attr("stroke","white") // clusterIndex === 'global' ? "#333" : color(clusterIndex))
    .attr("stroke-width",0.5);

  g.selectAll("text")
    .data(treemap.descendants().filter(d => d.depth > 0))
    .enter()
    .append("text")
    .attr("x", d => d.polygon.reduce((acc, point) => acc + point[0], 0) / d.polygon.length)
    .attr("y", d => d.polygon.reduce((acc, point) => acc + point[1], 0) / d.polygon.length)
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("font-size", d => clusterIndex === 'global' ? 
      Math.min(24, d.data[1] * 4000) + "px" :  // Larger font for global
      Math.min(20, d.data[1] * 800) + "px")
     .attr("fill", "white") // White text for all clusters
    //.attr("font-size", d => Math.min(20, d.data[1] * 1000) + "px")  // Adjusted scaling factor
     //.attr("fill", clusterIndex === 'global' ? "#333" : color(clusterIndex))
    //.attr("fill", clusterIndex === 'global' ? "#333" : "white") // Set color based on cluster type
    .text(d => d.data[0]);  // Display the theme word

  // Add hover-over tooltip
  const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

  // Update tooltip
  g.selectAll("path")
    .on("mouseover", function(event, d) {
      tooltip.transition()
        .duration(200)
        .style("opacity", .9);
      tooltip.html(`Theme: ${d.data[0]}<br/>Importance: ${d.data[1].toFixed(4)}`)
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

function createVisualization() {
  initializeVisualization();

  const topNWords = +document.getElementById("topNWords").value;
  const rotationAngle = +document.getElementById("rotationAngle").value || -40;

  const staticGroup = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  const rotatableGroup = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  const clusters = Object.keys(clusterData).filter(k => k !== 'global');
  console.log("Clusters:", clusters);
  
  const {pie, middleArc} = createDonutRings(clusters, staticGroup, volumeData);

  // Create Voronoi treemap for global cluster in the center
  if (clusterData['global']) {
    const clipPolygon = getClipPolygon(0, 2 * Math.PI, 0, innerRadius);
    const treemap = createVoronoiTreemap(clusterData['global'].slice(0, topNWords), clipPolygon, 'global');
    drawVoronoiTreemap(treemap, 0, 0, 'global', 'Global', staticGroup);
  }

  // Create Voronoi treemaps for each cluster
  pie(clusters).forEach((d, i) => {
    const cluster = clusterData[d.data];
    if (cluster) {
      const clipPolygon = getClipPolygon(d.startAngle, d.endAngle, middleRadius, outerRadius);
      const treemap = createVoronoiTreemap(cluster.slice(0, topNWords), clipPolygon, i);
      const centroid = middleArc.centroid(d);
      drawVoronoiTreemap(treemap, centroid[0], centroid[1], i, d.data, rotatableGroup);
    }
  });

  // Rotate only the treemaps
  rotatableGroup.attr("transform", `translate(${width/2},${height/2}) rotate(${rotationAngle})`);
}

// Set default values for the inputs
document.getElementById("topNWords").value = 100;
document.getElementById("innerRadiusStart").value = 38;
document.getElementById("innerRadiusEnd").value = 40;
document.getElementById("outerRadiusStart").value = 90;
document.getElementById("rotationAngle").value = -40;

// Load both data files
Promise.all([
  d3.json("proc_themes.json"),
  d3.json("cluster_volumes.json")
]).then(([themeData, loadedVolumeData]) => {
  clusterData = themeData;
  volumeData = loadedVolumeData;

  console.log("Processed cluster data:", clusterData);
  console.log("Cluster volumes (immediately after loading):", volumeData);

  // Call createVisualization in a separate tick to ensure data is fully processed
  setTimeout(createVisualization, 0);
}).catch(error => {
  console.error("Error loading data or drawing visualization:", error);
  alert("Error loading data. Please check the console for more information.");
});

