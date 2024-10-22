// Global configuration object
const VIZ_CONFIG = {
  // Dimensions
  width: 1400,
  height: 1400,
  
  // Font sizes
  fonts: {
    globalMin: 8,
    globalMax: 14,
    clusterMin: 8, 
    clusterMax: 20,
    clusterLabelSize: 12
  },
  
  // Word counts
  wordCounts: {
    global: 100,  // Words in center
    clusters: 50  // Words per outer cluster
  },
  
  // Visualization settings
  visualization: {
    padding: 80,
    innerRadiusStart: 38,
    innerRadiusEnd: 40,
    outerRadiusStart: 90,
    rotationAngle: -40
  },
  
  // Colors
  colorScheme: d3.schemeCategory10
};

// Global variables
let svg, width, height, radius, innerRadius, middleRadius, outerRadius, color;
let clusterData = {}, volumeData = {};

// SVG Download functionality
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

function getConfigFromUI() {
  return {
    ...VIZ_CONFIG,
    fonts: {
      globalMin: +document.getElementById("globalFontMin").value,
      globalMax: +document.getElementById("globalFontMax").value,
      clusterMin: +document.getElementById("clusterFontMin").value,
      clusterMax: +document.getElementById("clusterFontMax").value
    },
    wordCounts: {
      global: +document.getElementById("globalWords").value,
      clusters: +document.getElementById("clusterWords").value
    },
    visualization: {
      ...VIZ_CONFIG.visualization,
      innerRadiusStart: +document.getElementById("innerRadiusStart").value,
      innerRadiusEnd: +document.getElementById("innerRadiusEnd").value,
      outerRadiusStart: +document.getElementById("outerRadiusStart").value,
      rotationAngle: +document.getElementById("rotationAngle").value
    }
  };
}

function calculateFontSize(importance, isGlobal = false) {
  const config = getConfigFromUI();
  const { min, max } = isGlobal ? 
    { min: config.fonts.globalMin, max: config.fonts.globalMax } :
    { min: config.fonts.clusterMin, max: config.fonts.clusterMax };
    
  return Math.min(max, min + (importance * (max - min)));
}

function initializeVisualization() {
  const config = getConfigFromUI();
  
  svg = d3.select("#visualization");
  width = +svg.attr("width");
  height = +svg.attr("height");
  radius = Math.min(width, height) / 2 - config.visualization.padding;
  
  innerRadius = (config.visualization.innerRadiusStart / 100) * radius;
  middleRadius = (config.visualization.innerRadiusEnd / 100) * radius;
  outerRadius = (config.visualization.outerRadiusStart / 100) * radius;

  color = d3.scaleOrdinal(config.colorScheme);

  svg.selectAll("*").remove(); // Clear previous visualization
}

function createDonutRings(data, vizGroup, volumeData) {
  console.log("Creating donut rings with data:", data);
  const g = vizGroup.append("g");

  // Add white background for central area
  g.append("circle")
    .attr("r", innerRadius)
    .attr("fill", "white");

  const pie = d3.pie()
    .value(d => 1)
    .sort(null);

  const arc = d3.arc()
    .innerRadius(innerRadius)
    .outerRadius(middleRadius);

  // Inner ring (darker colors, cluster labels)
  g.selectAll("path.inner")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "inner")
    .attr("d", arc)
    .attr("fill", (d, i) => color(i))
    .attr("stroke", "white")
    .attr("stroke-width", 1);

  // Middle ring (pale, transparent background)
  const middleArc = d3.arc()
    .innerRadius(middleRadius)
    .outerRadius(outerRadius);

  g.selectAll("path.middle")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "middle")
    .attr("d", middleArc)
    .attr("fill", (d, i) => d3.color(color(i)).brighter(1.5))
    .attr("stroke", "none")
    .attr("opacity", 0.2);

  // Volume representation
  const volumes = Object.values(volumeData).filter(v => typeof v === 'number' && !isNaN(v));
  const localVolumes = volumes.slice(1); // Skip global volume

  const volumeScale = d3.scaleLinear()
    .domain([0, d3.max(localVolumes) * 1.2])
    .range([0, outerRadius - middleRadius]);

  g.selectAll("path.volume")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "volume")
    .attr("d", (d) => {
      const volume = volumeData[d.data] || 0;
      const outerRadiusAdjusted = middleRadius + volumeScale(volume);
      return d3.arc()
        .innerRadius(middleRadius)
        .outerRadius(outerRadiusAdjusted)(d);
    })
    .attr("fill", (d, i) => d3.color(color(i)).brighter(1.5))
    .attr("stroke", "none")
    .attr("opacity", 0.8);

  // Add cluster labels
  g.selectAll("text.cluster-label")
    .data(pie(data))
    .enter()
    .append("text")
    .attr("class", "cluster-label")
    .attr("transform", d => {
      const centroid = arc.centroid(d);
      return centroid[0] && centroid[1] ? `translate(${centroid[0]},${centroid[1]})` : "";
    })
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("fill", "white")
    .attr("font-size", `${VIZ_CONFIG.fonts.clusterLabelSize}px`)
    .attr("font-weight", "bold")
    .text((d) => volumeData[d.data] || 0);

  return {pie, middleArc};
}

function createVoronoiTreemap(data, clipPolygon, clusterIndex) {
  if (!data || !clipPolygon) {
    console.error("Invalid data or clipPolygon:", {data, clipPolygon});
    return null;
  }

  const voronoiTreemap = d3.voronoiTreemap()
    .clip(clipPolygon)
    .minWeightRatio(0.01)
    .prng(Math.random);

  const rootNode = d3.hierarchy({children: data})
    .sum(d => d[1]);

  try {
    voronoiTreemap(rootNode);
    return rootNode;
  } catch (e) {
    console.error("Error creating treemap:", e);
    return null;
  }
}

function drawVoronoiTreemap(treemap, x, y, clusterIndex, clusterName, vizGroup) {
  if (!treemap) return;

  const g = vizGroup.append("g")
    .attr("transform", `translate(${x || 0},${y || 0})`);

  // Add paths for treemap cells
  g.selectAll("path")
    .data(treemap.descendants().filter(d => d.depth > 0 && d.polygon))
    .enter()
    .append("path")
    .attr("d", d => `M${d.polygon.join("L")}Z`)
    .attr("fill", clusterIndex === 'global' ? "#333333" : color(clusterIndex))
    .attr("stroke", "white")
    .attr("stroke-width", 0.5);

  // Add text labels
  g.selectAll("text")
    .data(treemap.descendants().filter(d => d.depth > 0 && d.polygon))
    .enter()
    .append("text")
    .attr("x", d => {
      const x = d.polygon.reduce((acc, point) => acc + point[0], 0) / d.polygon.length;
      return isNaN(x) ? 0 : x;
    })
    .attr("y", d => {
      const y = d.polygon.reduce((acc, point) => acc + point[1], 0) / d.polygon.length;
      return isNaN(y) ? 0 : y;
    })
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("font-size", d => {
      const fontSize = calculateFontSize(d.data[1], clusterIndex === 'global');
      return `${fontSize}px`;
    })
    .attr("fill", "white")
    .text(d => d.data[0]);

  // Add tooltips
  const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

  g.selectAll("path")
    .on("mouseover", function(event, d) {
      tooltip.transition()
        .duration(200)
        .style("opacity", .9);
      tooltip.html(`Theme: ${d.data[0]}<br/>Importance: ${d.data[1].toFixed(4)}`)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
    })
    .on("mouseout", function() {
      tooltip.transition()
        .duration(500)
        .style("opacity", 0);
    });
}

function getClipPolygon(startAngle, endAngle, innerRadius, outerRadius) {
  if (isNaN(startAngle) || isNaN(endAngle) || isNaN(innerRadius) || isNaN(outerRadius)) {
    console.warn('Invalid parameters in getClipPolygon:', {startAngle, endAngle, innerRadius, outerRadius});
    return [[0,0]];
  }

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
  console.log("Starting visualization creation...");
  const config = getConfigFromUI();
  console.log("Current config:", config);
  
  initializeVisualization();

  const staticGroup = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  const rotatableGroup = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  const clusters = Object.keys(clusterData).filter(k => k !== 'global');
  console.log("Processing clusters:", clusters);
  
  const {pie, middleArc} = createDonutRings(clusters, staticGroup, volumeData);

  // Create global treemap
  if (clusterData['global']) {
    console.log("Creating global treemap");
    const clipPolygon = getClipPolygon(0, 2 * Math.PI, 0, innerRadius);
    const treemap = createVoronoiTreemap(
      clusterData['global'].slice(0, config.wordCounts.global), 
      clipPolygon, 
      'global'
    );
    if (treemap) {
      drawVoronoiTreemap(treemap, 0, 0, 'global', 'Global', staticGroup);
    }
  }

  // Create cluster treemaps
  pie(clusters).forEach((d, i) => {
    const cluster = clusterData[d.data];
    if (cluster) {
      console.log(`Creating treemap for cluster ${d.data}`);
      const clipPolygon = getClipPolygon(d.startAngle, d.endAngle, middleRadius, outerRadius);
      const treemap = createVoronoiTreemap(
        cluster.slice(0, config.wordCounts.clusters), 
        clipPolygon, 
        i
      );
      if (treemap) {
        const centroid = middleArc.centroid(d);
        if (centroid && !isNaN(centroid[0]) && !isNaN(centroid[1])) {
          drawVoronoiTreemap(treemap, centroid[0], centroid[1], i, d.data, rotatableGroup);
        }
      }
    }
  });

  rotatableGroup.attr("transform", `translate(${width/2},${height/2}) rotate(${config.visualization.rotationAngle})`);
}

// Event listener for visualization updates
document.getElementById("updateViz").addEventListener("click", createVisualization);

// Load data and initialize visualization
Promise.all([
  d3.json("proc_themes.json"),
  d3.json("cluster_volumes.json")
]).then(([themeData, volumeData]) => {
  clusterData = themeData;
  volumeData = volumeData;

  console.log("Loaded cluster data:", clusterData);
  console.log("Loaded volume data:", volumeData);

  setTimeout(createVisualization, 0);
}).catch(error => {
  console.error("Error loading data:", error);
  alert("Error loading data. Please check the console for details.");
});
