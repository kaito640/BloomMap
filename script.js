// Load the CSV data
// Load the CSV data
d3.csv("cluster_summary.csv").then(data => {

  // Prepare the SVG canvas
  const width = 960, height = 960;
  const radius = Math.min(width, height) / 2;
  
  const svg = d3.select("svg")
    .attr("width", width)
    .attr("height", height)
  .append("g")
    .attr("transform", `translate(${width / 2}, ${height / 2})`);
  
  // Group data by clusters
  const clusters = d3.group(data, d => d.cluster);
  
  // Create a scale for the angles
  const angleScale = d3.scaleLinear()
    .domain([0, clusters.size])
    .range([0, 2 * Math.PI]);
  
  const clusterRadius = radius / 3;  // To position clusters closer to the center
  
  // Create an arc generator
  const arc = d3.arc()
    .innerRadius(clusterRadius - 20)
    .outerRadius(clusterRadius + 20)
    .startAngle((d, i) => angleScale(i))
    .endAngle((d, i) => angleScale(i + 1));
  
  // Draw each cluster as an arc
  svg.selectAll(".cluster")
    .data(Array.from(clusters.keys()))
    .enter().append("path")
      .attr("class", "cluster")
      .attr("d", (d, i) => arc(d, i))
      .attr("fill", (d, i) => d3.schemeCategory10[i % 10])
      .attr("stroke", "#fff")
      .attr("stroke-width", 2);
  
  // Add text for each cluster centered on the arc
  svg.selectAll(".cluster-text")
    .data(Array.from(clusters.keys()))
    .enter().append("text")
      .attr("transform", (d, i) => {
        const [x, y] = arc.centroid(d, i);  // Get centroid of the arc
        return `translate(${x}, ${y})`;
      })
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .text(d => d)
      .style("font-size", "14px");  // Adjust the font size as needed
});

