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
    clusterLabelSize: 12,
    volumeLabelSize: 11
  },
  
  // Word counts
  wordCounts: {
    global: 100,  // Words in center
    clusters: 50  // Words per outer cluster
  },
  
  // Visualization settings
  visualization: {
    padding: 120,
    volumeScale: 1.2,
    innerRadiusStart: 38,
    innerRadiusEnd: 40,
    outerRadiusStart: 90,
    rotationAngle: -40
  },

  // Treemap settings
  treemap: {
    rotationOffset: 0,      // Additional rotation for treemap alignment
    radialOffset: 0,        // Offset from center point
    textRotationOffset: 0,  // Text rotation adjustment
    radialPosition: 0.9     // 0 = middle, 1 = outer
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
      clusterMax: +document.getElementById("clusterFontMax").value,
      volumeLabelSize: 11
    },
    wordCounts: {
      global: +document.getElementById("globalWords").value,
      clusters: +document.getElementById("clusterWords").value
    },
    visualization: {
      padding: +document.getElementById("vizPadding").value,
      volumeScale: +document.getElementById("volumeScale").value,
      innerRadiusStart: +document.getElementById("innerRadiusStart").value,
      innerRadiusEnd: +document.getElementById("innerRadiusEnd").value,
      outerRadiusStart: +document.getElementById("outerRadiusStart").value,
      rotationAngle: +document.getElementById("rotationAngle").value
    },
    treemap: {
      rotationOffset: +document.getElementById("treemapRotation").value || 0,
      radialOffset: +document.getElementById("treemapOffset").value || 0,
      textRotationOffset: +document.getElementById("textRotation").value || 0,
      radialPosition: +document.getElementById("radialPosition").value || 0.5
    }
  };
}

function calculateFontSize(importance, isGlobal = false) {
  const config = getConfigFromUI();
  const { min, max } = isGlobal ? 
    { min: config.fonts.globalMin, max: config.fonts.globalMax } :
    { min: config.fonts.clusterMin, max: config.fonts.clusterMax };
    
  // Using power scale for non-linear emphasis
  const scale = d3.scalePow()
    .exponent(0.5) // Square root scale for more emphasis on higher ranks
    .domain([0, 1])
    .range([min, max]);
    
  return scale(importance);
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
  console.log("Raw volume data:", volumeData);
  
  const g = vizGroup.append("g");

  // Add white background for central area
  g.append("circle")
    .attr("r", innerRadius)
    .attr("fill", "white");

  const pie = d3.pie()
    .value(d => 1)
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

  // Define volumeScale with robust global detection
  const volumes = Object.entries(volumeData)
    .filter(([key, value]) => 
      // Keep only entries with numeric values
      typeof value === 'number' && !isNaN(value)
    );
  
  console.log("Valid volume entries:", volumes);

  // Separate global and local volumes
  const globalVolume = volumes.find(([key, _]) => 
    key.toLowerCase() === 'global'
  )?.[1] || 0;

  const localVolumes = volumes
    .filter(([key, _]) => key.toLowerCase() !== 'global')
    .map(([_, value]) => value);

  console.log("Global volume:", globalVolume);
  console.log("Local volumes:", localVolumes);
  
  if (localVolumes.length === 0) {
    console.error("No valid local volume data found");
    return {pie, middleArc};
  }

  const volumeScale = d3.scaleLinear()
    .domain([0, d3.max(localVolumes) * 1.2])
    .range([0, outerRadius - middleRadius]);

  // Volume bars
  g.selectAll("path.volume")
    .data(pie(data))
    .enter()
    .append("path")
    .attr("class", "volume")
    .attr("d", (d) => {
      const volume = volumeData[d.data] || 0;
      console.log(`Volume for ${d.data}:`, volume);
      const outerRadiusAdjusted = middleRadius + volumeScale(volume);
      return arc
        .innerRadius(middleRadius)
        .outerRadius(outerRadiusAdjusted)
        .startAngle(d.startAngle + 0.02)
        .endAngle(d.endAngle - 0.02)
        (d);
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

  // Add cluster labels
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
      return `${volume}`;
    });

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

function drawVoronoiTreemap(treemap, x, y, clusterIndex, clusterName, vizGroup, angle) {
  if (!treemap) return;
 
  const rotation = clusterIndex === 'global' ? 0 : (angle * 180 / Math.PI) - 90;

  // Create a group for this treemap
  const g = vizGroup.append("g")
    .attr("transform", () => {
      if (clusterIndex === 'global') {
        return `translate(${x},${y})`;
      }
      
      // For peripheral treemaps:
      // 1. Translate to the label point
      // 2. Rotate to align with the panel angle
      //const rotation = (angle * 180 / Math.PI) - 90;
      //return `translate(${x},${y}) rotate(${rotation})`;
      return `translate(${x},${y})`;
    });

  // Add paths for treemap cells
  g.selectAll("path")
    .data(treemap.descendants().filter(d => d.depth > 0 && d.polygon))
    .enter()
    .append("path")
    .attr("d", d => `M${d.polygon.join("L")}Z`)
    .attr("fill", clusterIndex === 'global' ? "#333333" : color(clusterIndex))
    .attr("stroke", "white")
    .attr("stroke-width", 0.5);

  // Add text with rotation to maintain readability
  g.selectAll("text")
    .data(treemap.descendants().filter(d => d.depth > 0 && d.polygon))
    .enter()
    .append("text")
    .attr("transform", d => {
      const x = d.polygon.reduce((acc, point) => acc + point[0], 0) / d.polygon.length;
      const y = d.polygon.reduce((acc, point) => acc + point[1], 0) / d.polygon.length;
      //return `translate(${x},${y}) rotate(${-rotation})`;
      return `translate(${x},${y})`; // No rotation, text will stay horizontal
    })
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .attr("font-size", d => {
      const fontSize = calculateFontSize(d.data[1], clusterIndex === 'global');
      return `${fontSize}px`;
    })
    .attr("fill", "white")
    .text(d => d.data[0]);
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
  
  initializeVisualization();

  const staticGroup = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  const rotatableGroup = svg.append("g")
    .attr("transform", `translate(${width/2},${height/2})`);

  const clusters = Object.keys(clusterData).filter(k => k !== 'global');
  
  // Create the base arc for consistent measurements
  const arc = d3.arc()
    .innerRadius(innerRadius)
    .outerRadius(middleRadius);
    
  const pie = d3.pie()
    .value(d => 1)
    .sort(null);

  const pieData = pie(clusters);

  // Create donut rings first
  const {middleArc} = createDonutRings(clusters, staticGroup, volumeData);

  // Create global treemap
  if (clusterData['global']) {
    const clipPolygon = getClipPolygon(0, 2 * Math.PI, 0, innerRadius);
    const treemap = createVoronoiTreemap(
      clusterData['global'].slice(0, config.wordCounts.global), 
      clipPolygon, 
      'global'
    );
    if (treemap) {
      drawVoronoiTreemap(treemap, 0, 0, 'global', 'Global', staticGroup, 0);
    }
  }

  // Create cluster treemaps using the same angles as the panels
  pieData.forEach((d, i) => {
    const cluster = clusterData[d.data];
    if (cluster) {
      // Get the anchor point (label position)
      const radialPos = middleRadius + (outerRadius - middleRadius) * config.treemap.radialPosition;
      // Create an arc at this radius for centroid calculation
      const placementArc = d3.arc()
          .innerRadius(radialPos)
          .outerRadius(radialPos);
      // Get the anchor point at the adjusted radius
      const centroid = placementArc.centroid(d);

      const angle = (d.startAngle + d.endAngle) / 2;
      
      const clipPolygon = getClipPolygon(d.startAngle, d.endAngle, middleRadius, outerRadius);
      const treemap = createVoronoiTreemap(
        cluster.slice(0, config.wordCounts.clusters), 
        clipPolygon, 
        i
      );
      
      if (treemap) {
        drawVoronoiTreemap(
          treemap,
          centroid[0],
          centroid[1],
          i,
          d.data,
          rotatableGroup,
          angle
        );
      }
    }
  });

  // Apply final rotation to the whole group
  rotatableGroup.attr("transform", 
    `translate(${width/2},${height/2}) rotate(${config.visualization.rotationAngle})`
  );
}



// Event listener for visualization updates
document.getElementById("updateViz").addEventListener("click", createVisualization);

// Load data and initialize visualization
Promise.all([
  d3.json("proc_themes.json"),
  d3.json("cluster_volumes.json")
]).then(([themeData, loadedVolumeData]) => {
  console.log("Raw loaded volume data:", loadedVolumeData);
  
  // Ensure we're getting proper data
  if (!loadedVolumeData || typeof loadedVolumeData !== 'object') {
    console.error("Volume data not loaded properly");
    return;
  }

  clusterData = themeData;
  volumeData = loadedVolumeData;

  console.log("Processed cluster data:", clusterData);
  console.log("Volume data after assignment:", volumeData);

  setTimeout(createVisualization, 0);
}).catch(error => {
  console.error("Error loading data:", error);
  alert("Error loading data. Please check the console for details.");
});
