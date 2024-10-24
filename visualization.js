// Global state variables for visualization
let svg, width, height, radius, innerRadius, middleRadius, outerRadius, color;

// Add error styling
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
    rotationAngle: -35
  },

  // Treemap settings
  treemap: {
    rotationOffset: 0,
    radialOffset: 0,
    textRotationOffset: 0,
    radialPosition: 0.7
  },

  // Colors
  colorScheme: d3.schemeCategory10
};

const normalizeAngle = (angle) => {
    angle = angle % (2 * Math.PI);
    return angle < 0 ? angle + (2 * Math.PI) : angle;
};

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
    
  const scale = d3.scalePow()
    .exponent(0.5)
    .domain([0, 1])
    .range([min, max]);
    
  return scale(importance);
}

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

function cleanup() {
  window.removeEventListener('resize', handleResize);
  if (svg) {
    svg.selectAll("*").remove();
  }
}

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

function displayError(message) {
  const container = document.getElementById("visualization").parentElement;
  const errorDiv = document.createElement("div");
  errorDiv.className = "visualization-error";
  errorDiv.textContent = `Error: ${message}`;
  container.appendChild(errorDiv);
}

// [Your existing createDonutRings, createVoronoiTreemap, drawVoronoiTreemap, and getClipPolygon functions remain unchanged]

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
    
    const arc = d3.arc()
      .innerRadius(innerRadius)
      .outerRadius(middleRadius);
      
    const pie = d3.pie()
      .value(d => 1)
      .sort(null);

    const pieData = pie(clusters);

    const {middleArc} = createDonutRings(clusters, staticGroup, volumeData);

    // [Rest of your existing visualization code remains unchanged]
    
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
