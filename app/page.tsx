import Home from "./HomeClient";

export default function Page() {
  return <Home apiKey={process.env.API_KEY ?? ""} />;
}
