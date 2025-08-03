const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let detector;

// 👉 Update this to your CloudFront domain (NO trailing slash)
const CLOUDFRONT_BASE_URL = "https://d2winlq3m1yqc3.cloudfront.net/models";

const loadModel = async (modelName) => {
  const modelUrl = `${CLOUDFRONT_BASE_URL}/${modelName.toLowerCase().replace(".", "-")}/model.json`;

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: modelName,
      modelUrl: modelUrl
    }
  );

  console.log("✅ Model loaded:", modelName);
};

const runInference = async () => {
  const fileInput = document.getElementById("imageUpload");
  const modelType = document.getElementById("modelType").value;

  if (!fileInput.files.length) {
    alert("Please upload an image!");
    return;
  }

  await loadModel(modelType);

  const image = new Image();
  const reader = new FileReader();

  reader.onload = async function (event) {
    image.src = event.target.result;

    image.onload = async () => {
      canvas.width = image.width;
      canvas.height = image.height;
      ctx.drawImage(image, 0, 0);

      const poses = await detector.estimatePoses(image);

      drawPoses(poses);
      console.log("🧠 Inference complete", poses);

      // Optional: Send image to backend
      // await uploadToBackend(image.src, modelType);
    };
  };

  reader.readAsDataURL(fileInput.files[0]);
};

const drawPoses = (poses) => {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "lime";
  ctx.fillStyle = "red";

  poses.forEach((pose) => {
    pose.keypoints.forEach((keypoint) => {
      if (keypoint.score > 0.3) {
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    drawSkeleton(pose.keypoints);
  });
};

const drawSkeleton = (keypoints) => {
  const adjacentPairs = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);

  adjacentPairs.forEach(([i, j]) => {
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    if (kp1.score > 0.3 && kp2.score > 0.3) {
      ctx.beginPath();
      ctx.moveTo(kp1.x, kp1.y);
      ctx.lineTo(kp2.x, kp2.y);
      ctx.stroke();
    }
  });
};

// Optional: send the image and selected model to your backend
/*
const uploadToBackend = async (base64Image, modelType) => {
  const response = await fetch("https://your-backend-url/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: base64Image,
      modelType: modelType
    })
  });

  const result = await response.json();
  console.log("✅ Backend Response:", result);
};
*/
