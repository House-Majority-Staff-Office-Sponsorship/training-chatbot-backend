export const metadata = {
  title: "Training Chatbot Backend",
  description: "RAG-powered Gemini API backend",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
