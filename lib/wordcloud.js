// Load the global summary CSV to get the top words
d3.csv("global_summary.csv").then(globalData => {

  // Logging the loaded data
  console.log("Global Summary Data Loaded:", globalData);

  // Create a canvas for the word cloud
  const canvasWidth = 600, canvasHeight = 600;
  const canvas = document.createElement("canvas");
  document.body.appendChild(canvas);

  canvas.width = canvasWidth;
  canvas.height = canvasHeight;

  // Limit number of words for the word cloud (top 50 most frequent words)
  const topXWords = globalData.slice(0, 50).map(d => ({ text: d.word, size: +d.frequency }));

  console.log("Top Words for Word Cloud:", topXWords);

  // Calculate the radius for the circular bounding box
  const radius = Math.min(canvasWidth, canvasHeight) / 2;

  // Custom function to keep words within a circle
  function withinCircle(x, y, r) {
    const dx = x - canvasWidth / 2;  // Distance from center (x-axis)
    const dy = y - canvasHeight / 2; // Distance from center (y-axis)
    return (dx * dx + dy * dy) <= (r * r); // Check if within radius
  }

  // Define the word cloud layout with circular bounding
  const layout = d3.layout.cloud()
    .size([canvasWidth, canvasHeight])
    .words(topXWords)
    .padding(5)
    .fontSize(d => Math.log(d.size) * 10)  // Scale font size based on frequency
    .rotate(() => ~~(Math.random() * 5) - 2)  // Rotation between -17° and 17°
    .spiral("archimedean")  // Use default spiral
    .on("word", function(d) {
      // Override word placement to check if it fits within the circle
      let attempts = 0;
      do {
        d.x = (Math.random() * canvasWidth * 2) ;
        d.y = (Math.random() * canvasHeight* 2) ;
        attempts++;
      } while (!withinCircle(d.x, d.y, radius) && attempts < 50);
    })
    .on("end", drawWordCloud);

  layout.start();

  function drawWordCloud(words) {
    const ctx = canvas.getContext("2d");

    // Clear canvas before drawing
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Translate the entire canvas context to center
    ctx.translate(canvasWidth / 2, canvasHeight / 2);

    words.forEach(d => {
      ctx.save();
      ctx.translate(d.x - canvasWidth / 2, d.y - canvasHeight / 2);  // Translate each word relative to canvas center
      ctx.rotate(d.rotate * Math.PI / 180);
      ctx.font = `${d.size}px Arial`;
      ctx.fillStyle = d3.schemeCategory10[Math.floor(Math.random() * 10)];
      ctx.textAlign = "center";
      ctx.fillText(d.text, 0, 0);
      ctx.restore();
    });
  }

}).catch(error => console.error("Error loading data or drawing word cloud:", error));

