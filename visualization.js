// Global state variables for visualization
let svg, width, height, radius, innerRadius, middleRadius, outerRadius, color;

// Add error styling for user feedback
const style = document.createElement('style');
style.textContent = `.visualization-error {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  background: #fee; color: #c00; padding: 1em; border-radius: 4px; border: 1px solid #fcc; font-family: sans-serif;
}`;
document.head.appendChild(style);

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
    padding: 240,
    volumeScale: 1.2,
    innerRadiusStart: 33,
    innerRadiusEnd: 40,
    outerRadiusStart: 100,
    rotationAngle: 28  // Default rotation angle
  },

  // Treemap settings
  treemap: {
    // rotationOffset: 0,      // Deprecated
    // radialOffset: 0,        // Deprecated
    textRotationOffset: 0,  // Deprecated
    radialPosition: 0.7  // Controls flower openness (0 = closed, 1 = fully open)
  },

  // Colors
  colorScheme: d3.schemeCategory10
};

// Normalize angles to positive values within 2π
const normalizeAngle = (angle) => {
    angle = angle % (2 * Math.PI);
    return angle < 0 ? angle + (2 * Math.PI) : angle;
};

// Get configuration from UI controls
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
      radialPosition: +document.getElementById("radialPosition").value || 0.7
    }
  };
}

// Helper function to rotate a point around origin
function rotatePoint(x, y, angleInDegrees) {
  const angleInRadians = (angleInDegrees * Math.PI) / 180;
  return {
    x: x * Math.cos(angleInRadians) - y * Math.sin(angleInRadians),
    y: x * Math.sin(angleInRadians) + y * Math.cos(angleInRadians)
  };
}

// Calculate font size based on importance and context
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

// Initialize or reinitialize the visualization
function initializeVisualization() {
  try {
    const config = getConfigFromUI();
    
    // Clean up existing visualization
    if (svg) {
      svg.selectAll("*").remove();
    }
    
    svg = d3.select("#visualization")
      .attr("width", config.width)
      .attr("height", config.height);
      
    width = config.width;
    height = config.height;
    radius = Math.min(width, height) / 2 - config.visualization.padding;
    
    // Validate radii
    if (config.visualization.innerRadiusStart >= config.visualization.innerRadiusEnd ||
        config.visualization.innerRadiusEnd >= config.visualization.outerRadiusStart) {
      throw new Error("Invalid radius configuration");
    }
    
    innerRadius = (config.visualization.innerRadiusStart / 100) * radius;
    middleRadius = (config.visualization.innerRadiusEnd / 100) * radius;
    outerRadius = (config.visualization.outerRadiusStart / 100) * radius;

    // Ensure color scheme exists
    color = d3.scaleOrdinal(config.colorScheme || d3.schemeCategory10);
    
    // Add resize handler
    window.removeEventListener('resize', handleResize);
    window.addEventListener('resize', handleResize);
    
  } catch (error) {
    console.error("Visualization initialization failed:", error);
    displayError(error.message);
    throw error;
  }
}

// Cleanup function for removing event listeners and clearing SVG
function cleanup() {
  window.removeEventListener('resize', handleResize);
  if (svg) {
    svg.selectAll("*").remove();
  }
}

// Handle window resize events
function handleResize() {
  const container = document.getElementById("visualization").parentElement;
  const newWidth = container.clientWidth;
  const newHeight = container.clientHeight;
  
  if (newWidth !== width || newHeight !== height) {
    width = newWidth;
    height = newHeight;
    createVisualization();
  }
}

// Validate input data structure
function validateData(themeData, volumeData) {
  if (!themeData || !volumeData) {
    throw new Error("Missing required data");
  }
  
  if (!themeData.global || !Array.isArray(themeData.global)) {
    throw new Error("Invalid global theme data format");
  }
  
  const clusters = Object.keys(themeData).filter(k => k !== 'global');
  if (clusters.length === 0) {
    throw new Error("No cluster data found");
  }
  
  clusters.forEach(cluster => {
    if (!Array.isArray(themeData[cluster])) {
      throw new Error(`Invalid data format for cluster: ${cluster}`);
    }
    if (!volumeData[cluster] || isNaN(volumeData[cluster])) {
      throw new Error(`Missing or invalid volume data for cluster: ${cluster}`);
    }
  });
}

// Display error messages to user
function displayError(message) {
  const container = document.getElementById("visualization").parentElement;
  const errorDiv = document.createElement("div");
  errorDiv.className = "visualization-error";
  errorDiv.textContent = `Error: ${message}`;
  container.appendChild(errorDiv);
}

// Create donut rings structure
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
    .filter(([key, value]) => typeof value === 'number' && !isNaN(value));
  
  console.log("Valid volume entries:", volumes);

  // Separate global and local volumes
  const globalVolume = volumes.find(([key, _]) => 
    key.toLowerCase() === 'global'
  )?.[1] || 0;

  const localVolumes = volumes
    .filter(([key, _]) => key.toLowerCase() !== 'global')
    .map(([_, value]) => value);

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

  // Cluster name labels near inner ring
  g.selectAll("text.cluster-label")
    .data(pie(data))
    .enter()
    .append("text")
    .attr("class", "cluster-label")
    .attr("transform", d => {
      const angle = normalizeAngle((d.startAngle + d.endAngle) / 2 - Math.PI / 2);
      const radius = innerRadius + 15;
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      return `translate(${x},${y})`;
    })
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "middle")
    .attr("fill", "white")
    .attr("font-size", "12px")
    .attr("font-weight", "bold")
    .text(d => d.data);

  // Add volume count labels
  g.selectAll("text.volume-label")
    .data(pie(data))
    .enter()
    .append("text")
    .attr("class", "volume-label")
    .attr("transform", d => {
      const volume = volumeData[d.data] || 0;
      const angle = (d.startAngle + d.endAngle) / 2;
      const outerRadiusAdjusted = middleRadius + volumeScale(volume);
      const x = Math.cos(angle - Math.PI / 2) * outerRadiusAdjusted;
      const y = Math.sin(angle - Math.PI / 2) * outerRadiusAdjusted;
      return `translate(${x},${y})`;
    })
    .attr("text-anchor", d => {
      const angle = (d.startAngle + d.endAngle) / 2;
      if (angle < Math.PI * 0.25 || angle > Math.PI * 1.75) return "start";
      if (angle >= Math.PI * 0.75 && angle <= Math.PI * 1.25) return "end";
      return "middle";
    })
    .attr("dominant-baseline", d => {
      const angle = (d.startAngle + d.endAngle) / 2;
      return angle < Math.PI ? "baseline" : "hanging";
    })
    .attr("fill", "black")
    .attr("font-size", "10px")
    .text(d => volumeData[d.data] || 0);

  return {pie, middleArc};
}

// Create Voronoi treemap
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

// Draw Voronoi treemap with enhanced text rotation
function drawVoronoiTreemap(treemap, x, y, clusterIndex, clusterName, vizGroup, angle) {
  if (!treemap) return;
  
  const config = getConfigFromUI();
  
  // Calculate base rotation for the cluster
  const clusterRotation = clusterIndex === 'global' ? 0 : (angle * 180 / Math.PI - 90); 
  
  // Create group for this treemap
  const g = vizGroup.append("g")
    .attr("transform", `translate(${x},${y})`);

  // Add paths for treemap cells
  g.selectAll("path")
    .data(treemap.descendants().filter(d => d.depth > 0 && d.polygon))
    .enter()
    .append("path")
    .attr("d", d => `M${d.polygon.join("L")}Z`)
    .attr("fill", clusterIndex === 'global' ? "#333333" : color(clusterIndex))
    .attr("stroke", "white")
    .attr("stroke-width", 0.5);

// Add text with enhanced rotation calculation
  g.selectAll("text")
    .data(treemap.descendants().filter(d => d.depth > 0 && d.polygon))
    .enter()
    .append("text")
    .attr("transform", d => {
      // Calculate centroid of the polygon for text placement
      const x = d.polygon.reduce((acc, point) => acc + point[0], 0) / d.polygon.length;
      const y = d.polygon.reduce((acc, point) => acc + point[1], 0) / d.polygon.length;
      
      // Calculate text rotation to maintain horizontal text
      // Counter-rotate against both cluster angle and overall visualization rotation
      //const textRotation = clusterIndex === 'global' ? 
      //  -config.visualization.rotationAngle : 
      //  -(clusterRotation + config.visualization.rotationAngle);

      // Counter-rotate against the overall visualization rotation
      const globalRotation = config.visualization.rotationAngle;


      // For non-global clusters, rotate text to match sector angle

      if (clusterIndex !== 'global') {
        return `translate(${x},${y}) rotate(${globalRotation})`;
      } else {
        return `translate(${x},${y})`;
      }

return `translate(${x},${y}) rotate(${-globalRotation})  `;
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

// Get clip polygon for treemap boundaries
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

// Main visualization creation function
function createVisualization() {
  try {
    console.log("Starting visualization creation...");
    validateData(clusterData, volumeData);
    
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

    // Create cluster treemaps
    pieData.forEach((d, i) => {
      const cluster = clusterData[d.data];
      if (cluster) {
        const radialPos = middleRadius + (outerRadius - middleRadius) * config.treemap.radialPosition;
        const placementArc = d3.arc()
          .innerRadius(radialPos)
          .outerRadius(radialPos);
	
        // Get initial centroid
	const centroid = placementArc.centroid(d);
	// Pre-rotate the centroid by the compensation angle
        const globalRotation = config.visualization.rotationAngle;
        const rotationCompensation = -1*(globalRotation) 
        const rotatedCentroid = rotatePoint(centroid[0], centroid[1], rotationCompensation);        

        const angle = (d.startAngle + d.endAngle) / 2;
        
        const clipPolygon = getClipPolygon(d.startAngle, d.endAngle, middleRadius, outerRadius);
        // Also rotate the clip polygon points
        const rotatedClipPolygon = clipPolygon.map(point => {
          const rotated = rotatePoint(point[0], point[1], rotationCompensation);
	  return [rotated.x, rotated.y];
        });        


	const treemap = createVoronoiTreemap(
          cluster.slice(0, config.wordCounts.clusters), 
          rotatedClipPolygon, 
          i
        );

        if (treemap) {
          drawVoronoiTreemap(
            treemap,
            rotatedCentroid.x,
            rotatedCentroid.y,
            i,
            d.data,
            rotatableGroup,
            angle
          );
        }
      }
    });

    // Make all treemap text horizontal
    rotatableGroup.selectAll("text")
      .attr("transform", function() {
        const transform = d3.select(this).attr("transform");
        const x = transform.match(/translate\(([\d.-]+),([\d.-]+)\)/);
        if (x) {
          return `translate(${x[1]},${x[2]}) rotate(0)`;  // Keep translation, remove any rotation
        }
        return transform;
      });

    // Apply final rotation to the whole group
    rotatableGroup.attr("transform", 
      `translate(${width/2},${height/2})`
    );
    
  } catch (error) {
    console.error("Failed to create visualization:", error);
    displayError(error.message);
  }
}

// Global state
let clusterData = {}, volumeData = {};

// Event listeners
window.addEventListener('beforeunload', cleanup);

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

// Update visualization button
document.getElementById("updateViz").addEventListener("click", () => {
  createVisualization();
});


// Load data and initialize visualization
Promise.all([
  d3.json("proc_themes.json"),
  d3.json("cluster_volumes.json")
]).then(([themeData, loadedVolumeData]) => {
  try {
    validateData(themeData, loadedVolumeData);
    clusterData = themeData;
    volumeData = loadedVolumeData;
    setTimeout(createVisualization, 0);
  } catch (error) {
    console.error("Data validation failed:", error);
    displayError(error.message);
  }
}).catch(error => {
  console.error("Error loading data:", error);
  displayError("Failed to load visualization data");
});






